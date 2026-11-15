"""
Versiones B y C: modelo two-tower (torre de producto + torre de cliente).

Entrada:  data/processed/transacciones.csv, data/processed/catalogo.csv,
          data/embeddings/texto.npz  (B y C), data/embeddings/imagen.npz  (solo C)
Salida:   data/modelos/version_{B,C}.pt            pesos de la mejor corrida + config
          reports/resultados/version_{B,C}.csv     mismo formato que version_A.csv
          reports/resultados/iteraciones.csv       una fila por corrida (reemplaza las filas de la versión)
          reports/resultados/ablacion.csv          variantes (A, pop+recompra, puro, +pop, +pop+recompra)
          reports/figuras/curva_{B,C}.png          loss de train y recall@12 de val por época
          reports/figuras/ejemplos_B.png           3 clientes de test con sus recomendaciones
          --transfer: reports/resultados/transfer_{B,C}.csv  completo vs base sin entrenar vs sin preentrenados

Modelo:
  Torre de producto: concat[texto (fijo), embeddings aprendidos de seccion, tipo_prenda,
    color_base y rango de precio, (C: imagen CLIP fija), (it4: embedding de id del item)]
    -> Linear -> ReLU -> Linear(d) -> L2.
  Torre de cliente: promedio de las salidas de la torre de producto de sus últimas 30 compras
    (pesos por recencia) -> Linear -> ReLU -> Linear(d) -> L2.
  Pérdida: softmax contrastivo con negativos dentro del batch (tau = 0.05), con corrección logQ
    (logits - log q_j, q = frecuencia del item entre los positivos) y máscara de falsos negativos.
Entrenamiento: pares (cliente, item) de las últimas N semanas de train (4 en it1, 12 después);
  la historia es lo comprado antes de esa semana.
Recomendación: puntaje = cos(u, i) + alpha * z(log(1 + compras de la semana previa)) + beta * r,
  con r(c, i) = suma de exp(-días_atrás / 30) sobre las compras de c de ese item (recompra, es una
  regla, no deep learning). (alpha, beta) se eligen en val.
Iteraciones (acumulativas, d=64 lr=1e-3 batch=512): it1 base, it2 +logQ, it3 +12 semanas,
  it4 +id item; it5 = rejilla de 8 configs sobre la mejor iteración (por recall@12 de val).

Uso:
  python src/models/two_tower.py            (versión B)
  python src/models/two_tower.py --imagen   (versión C)
  python src/models/two_tower.py --transfer [--imagen]   (solo transfer learning; no toca los demás archivos)
    Compara (puro y + pop + recompra): modelo completo (checkpoint), base zero-shot (coseno entre embeddings
    fijos, sin entrenar) y two-tower entrenado sin embeddings preentrenados (misma config ganadora).
    Con --imagen agrega "C con config de B": texto + imagen entrenado con la config y opciones de version_B.pt.
  Opciones: --epocas N (máximo de épocas), --una (it5 solo con la primera config de la rejilla)
"""

import argparse
import itertools
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.metricas import evaluar  # noqa: E402
from models.baseline import mas_comprados, seccion_cliente, fila, SEMANA_PREVIA, K  # noqa: E402

TRANSACCIONES = "data/processed/transacciones.csv"
CATALOGO = "data/processed/catalogo.csv"
TEXTO = "data/embeddings/texto.npz"
IMAGEN = "data/embeddings/imagen.npz"
ITERACIONES = "reports/resultados/iteraciones.csv"
ABLACION = "reports/resultados/ablacion.csv"
VERSION_A = "reports/resultados/version_A.csv"
FECHA0 = pd.Timestamp("2020-03-24")
FIN_TRAIN = pd.Timestamp("2020-09-08")
CORTE = {"val": pd.Timestamp("2020-09-09"), "test": pd.Timestamp("2020-09-16")}

SEMILLA = 42
TAU = 0.05
N_HIST = 30        # compras recientes que entran a la torre de cliente
VIDA_DIAS = 30     # peso de una compra = exp(-dias_atras / VIDA_DIAS)
SEMANAS_TRAIN = 12
DIM_ID = 32        # embedding de id del item (it4)
DIM_CAT = 16       # tamaño de cada embedding aprendido
OCULTA = 256
MAX_EPOCAS = 20
PACIENCIA = 3
ALFAS = [0, 0.05, 0.1, 0.2, 0.5, 1]
BETAS = [0, 0.1, 0.2, 0.5, 1, 2]
PARES = list(itertools.product(ALFAS, BETAS))
BASE = (64, 1e-3, 512)  # d, lr, batch de las iteraciones it1-it4
ITERS = [("it1 base", dict(logq=False, semanas=4, id=False)),
         ("it2 +logQ", dict(logq=True, semanas=4, id=False)),
         ("it3 +12 semanas", dict(logq=True, semanas=SEMANAS_TRAIN, id=False)),
         ("it4 +id item", dict(logq=True, semanas=SEMANAS_TRAIN, id=True))]
REJILLA = list(itertools.product([64, 128], [1e-3, 3e-4], [512, 2048]))  # d, lr, batch
CATEGORICAS = ["seccion", "tipo_prenda", "color_base", "rango_precio"]
device = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------- datos ----------

def cargar(imagen):
    """Catálogo, transacciones (con índices enteros) y embeddings fijos alineados."""
    cat = pd.read_csv(CATALOGO, dtype={"item_id": str})
    tx = pd.read_csv(TRANSACCIONES, dtype={"item_id": str, "customer_id": str})
    cat["rango_precio"] = pd.qcut(cat["precio_q"].rank(method="first"), 5, labels=False)
    vocab = {c: {v: i for i, v in enumerate(sorted(cat[c].unique()))} for c in CATEGORICAS}
    cats = np.stack([cat[c].map(vocab[c]).values for c in CATEGORICAS], axis=1)

    z = np.load(TEXTO, allow_pickle=True)
    assert list(z["item_id"]) == list(cat["item_id"]), "texto.npz desalineado con el catálogo"
    fijos = [z["emb"]]
    if imagen:
        zi = np.load(IMAGEN, allow_pickle=True)
        assert list(zi["item_id"]) == list(cat["item_id"]), "imagen.npz desalineado con el catálogo"
        fijos.append(zi["emb"])

    tx = tx[tx["item_id"].isin(set(cat["item_id"]))].copy()
    tx["ii"] = tx["item_id"].map({v: i for i, v in enumerate(cat["item_id"])})
    tx["cc"], clientes = pd.factorize(tx["customer_id"])
    tx["day"] = (pd.to_datetime(tx["t_dat"]) - FECHA0).dt.days
    return cat, tx, np.concatenate(fijos, axis=1).astype(np.float32), cats, vocab, clientes


def dia(fecha):
    return (fecha - FECHA0).days


def historial(tx, n_clientes, corte):
    """Últimas N_HIST compras (item, días atrás) de cada cliente antes de `corte` (día). Pad = -1."""
    h = tx[tx["day"] < corte].sort_values(["cc", "day"], kind="stable")
    rango = h.groupby("cc").cumcount(ascending=False)
    h, rango = h[rango < N_HIST], rango[rango < N_HIST]
    items = np.full((n_clientes, N_HIST), -1, dtype=np.int64)
    dias = np.zeros((n_clientes, N_HIST), dtype=np.float32)
    col = N_HIST - 1 - rango.values
    items[h["cc"].values, col] = h["ii"].values
    dias[h["cc"].values, col] = corte - h["day"].values
    return items, dias


def ejemplos_train(tx, n_clientes, semanas):
    """Pares únicos (cliente, item) de cada una de las últimas semanas de train + su historia."""
    hi, hd, pos = [], [], []
    for s in range(semanas):
        fin = dia(FIN_TRAIN) - 7 * s
        ini = fin - 6
        items, dias = historial(tx, n_clientes, ini)
        par = tx[(tx["day"] >= ini) & (tx["day"] <= fin)][["cc", "ii"]].drop_duplicates()
        ok = (items[par["cc"].values] >= 0).any(axis=1)  # descarta clientes sin historial
        par = par[ok]
        hi.append(items[par["cc"].values]); hd.append(dias[par["cc"].values]); pos.append(par["ii"].values)
    t = lambda a: torch.from_numpy(np.concatenate(a)).to(device)
    return t(hi), t(hd), t(pos)


# ---------- modelo ----------

class TwoTower(nn.Module):
    def __init__(self, fijos, cats, tamanos, d, id_item=False):
        super().__init__()
        self.register_buffer("fijos", torch.from_numpy(fijos), persistent=False)
        self.register_buffer("cats", torch.from_numpy(cats), persistent=False)
        self.embs = nn.ModuleList([nn.Embedding(n, DIM_CAT) for n in tamanos])
        self.id = nn.Embedding(len(fijos), DIM_ID) if id_item else None
        entrada = fijos.shape[1] + DIM_CAT * len(tamanos) + (DIM_ID if id_item else 0)
        self.producto = nn.Sequential(nn.Linear(entrada, OCULTA), nn.ReLU(), nn.Linear(OCULTA, d))
        self.cliente = nn.Sequential(nn.Linear(d, OCULTA), nn.ReLU(), nn.Linear(OCULTA, d))

    def items(self):
        """Embeddings L2 de todos los items del catálogo."""
        x = [self.fijos] + [e(self.cats[:, j]) for j, e in enumerate(self.embs)]
        if self.id is not None:
            x.append(self.id.weight)
        return F.normalize(self.producto(torch.cat(x, dim=1)), dim=1)

    def clientes(self, I, hist, dias):
        """Embeddings L2 de clientes: promedio ponderado por recencia de sus compras recientes."""
        w = torch.exp(-dias / VIDA_DIAS) * (hist >= 0)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        prom = (w.unsqueeze(-1) * I[hist.clamp(min=0)]).sum(1)
        return F.normalize(self.cliente(prom), dim=1)


def perdida(u, i_pos, pos, logq=None):
    """Softmax contrastivo con negativos del batch; enmascara falsos negativos (mismo item).
    logq (opcional, [n_items]): log de la frecuencia de cada item, se resta a los logits."""
    logits = u @ i_pos.T / TAU
    if logq is not None:
        logits = logits - logq[pos].unsqueeze(0)
    mismo = (pos.unsqueeze(0) == pos.unsqueeze(1)) & ~torch.eye(len(pos), dtype=torch.bool, device=pos.device)
    logits = logits.masked_fill(mismo, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(pos), device=pos.device))


# ---------- recomendación y evaluación ----------

def z_popularidad(tx, n_items, split):
    ini, fin = SEMANA_PREVIA[split]
    sem = tx[(tx["t_dat"] >= ini) & (tx["t_dat"] <= fin)]
    l = np.log1p(np.bincount(sem["ii"].values, minlength=n_items))
    return torch.tensor((l - l.mean()) / l.std(), dtype=torch.float32, device=device)


@torch.no_grad()
def top_items(modelo, hist, dias, z, pares, bloque=2048):
    """Para cada (alpha, beta), los K items con mayor puntaje por cliente -> dict par -> array [n, K].
    Puntaje = cos(u, i) + alpha * z + beta * r. modelo=None: sin coseno (pop + recompra)."""
    if modelo is not None:
        modelo.eval()
        I = modelo.items()
    h, d_ = torch.from_numpy(hist).to(device), torch.from_numpy(dias).to(device)
    salida = {p: [] for p in pares}
    for b in range(0, len(h), bloque):
        hb, db = h[b:b + bloque], d_[b:b + bloque]
        w = torch.exp(-db / VIDA_DIAS) * (hb >= 0)
        r = torch.zeros(len(hb), len(z), device=device).scatter_add_(1, hb.clamp(min=0), w)
        s = modelo.clientes(I, hb, db) @ I.T if modelo is not None else 0
        for a, be in pares:
            salida[(a, be)].append((s + a * z + be * r).topk(K, dim=1).indices.cpu().numpy())
    return {p: np.concatenate(v) for p, v in salida.items()}


class Evaluador:
    """Evalúa val/test: clientes con historial -> modelo; sin historial -> top 12 de popularidad."""

    def __init__(self, cat, tx, clientes):
        self.cat, self.tx, self.clientes = cat, tx, clientes
        self.ids = cat["item_id"].values
        cod = {c: i for i, c in enumerate(clientes)}
        self.d = {}
        for split in ["val", "test"]:
            reales = tx[tx["split"] == split].groupby("customer_id")["item_id"].agg(set).to_dict()
            hist, dias = historial(tx, len(clientes), dia(CORTE[split]))
            con = [c for c in reales if c in cod and (hist[cod[c]] >= 0).any()]
            filas = np.array([cod[c] for c in con], dtype=int)
            top = mas_comprados(tx, *SEMANA_PREVIA[split])
            self.d[split] = dict(reales=reales, con=con, hist=hist[filas], dias=dias[filas],
                                 pop={c: top for c in reales}, z=z_popularidad(tx, len(cat), split))

    def recomendar(self, modelo, split, pares):
        """dict (alpha, beta) -> {cliente: lista de item_id}."""
        e = self.d[split]
        tops = top_items(modelo, e["hist"], e["dias"], e["z"], pares)
        res = {}
        for p, t in tops.items():
            rec = dict(e["pop"])
            rec.update({c: list(self.ids[t[k]]) for k, c in enumerate(e["con"])})
            res[p] = rec
        return res

    def metricas(self, modelo, split, pares):
        rec = self.recomendar(modelo, split, pares)
        return {p: evaluar(r, self.d[split]["reales"], K) for p, r in rec.items()}


def elegir(modelo, ev, pares=PARES):
    """(alpha, beta) con mayor recall@12 de val entre `pares`; devuelve (par, métricas val de todos)."""
    val = ev.metricas(modelo, "val", pares)
    return max(pares, key=lambda p: val[p][f"recall@{K}"]), val


# ---------- entrenamiento ----------

def entrenar(cfg, opc, data, ev, tamanos, max_epocas, semilla=SEMILLA):
    d, lr, batch = cfg
    cat, tx, fijos, cats, vocab, clientes = data
    torch.manual_seed(semilla); np.random.seed(semilla)
    modelo = TwoTower(fijos, cats, tamanos, d, opc["id"]).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    hi, hd, pos = ejemplos_train(tx, len(clientes), opc["semanas"])
    n = len(pos)
    logq = None
    if opc["logq"]:
        q = torch.bincount(pos, minlength=len(cat)).float().clamp(min=1)  # solo se usan items con q > 0
        logq = torch.log(q / q.sum())
    gen = torch.Generator().manual_seed(semilla)
    mejor, mejor_ep, estado, hist_loss, hist_rec = -1, 0, None, [], []
    t0 = time.time()
    for ep in range(1, max_epocas + 1):
        modelo.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total, nb = 0.0, 0
        for b in range(0, n - batch + 1, batch):  # el último batch incompleto se descarta
            idx = perm[b:b + batch]
            I = modelo.items()
            loss = perdida(modelo.clientes(I, hi[idx], hd[idx]), I[pos[idx]], pos[idx], logq)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        rec = ev.metricas(modelo, "val", [(0, 0)])[(0, 0)][f"recall@{K}"]
        hist_loss.append(total / nb); hist_rec.append(rec)
        print(f"  ép {ep:>2}  loss {total / nb:.4f}  val recall@12 puro {rec:.4f}", flush=True)
        if rec > mejor:
            mejor, mejor_ep = rec, ep
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        elif ep - mejor_ep >= PACIENCIA:
            break
    modelo.load_state_dict(estado)
    return modelo, dict(epocas=ep, mejor_epoca=mejor_ep, loss=hist_loss, recall=hist_rec,
                        segundos=time.time() - t0, n_ejemplos=n)


# ---------- salidas ----------

def tabla_resultados(ev, modelo, par):
    """Filas val/test (todos) y test por sección, como version_A.csv."""
    secciones = seccion_cliente(ev.tx, ev.cat)
    filas = []
    for split in ["val", "test"]:
        rec = ev.recomendar(modelo, split, [par])[par]
        reales = ev.d[split]["reales"]
        filas.append(fila(split, "todos", rec, reales))
        if split == "test":
            grupos = pd.Series({c: secciones.get(c, "sin historial") for c in reales})
            for s in sorted(grupos.unique()):
                reales_s = {c: reales[c] for c in grupos.index[grupos == s]}
                filas.append(fila(split, f"seccion={s}", rec, reales_s))
    return pd.DataFrame(filas)


def curva(info, version, d, lr, batch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = range(1, len(info["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ep, info["loss"], "o-", color="tab:blue", label="loss train")
    ax.set_xlabel("época"); ax.set_ylabel("loss (train)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(ep, info["recall"], "s-", color="tab:orange", label="recall@12 val")
    ax2.axvline(info["mejor_epoca"], ls="--", color="gray")
    ax2.set_ylabel("recall@12 val (puro)", color="tab:orange")
    ax.set_title(f"Versión {version}: d={d}, lr={lr}, batch={batch}")
    fig.tight_layout()
    os.makedirs("reports/figuras", exist_ok=True)
    fig.savefig(f"reports/figuras/curva_{version}.png", dpi=150)
    plt.close(fig)


def ejemplos(ev, modelo, par, n=3, n_compras=5, n_rec=6):
    """3 clientes de test con historial: últimas 5 compras y 6 recomendaciones (verde = acertó)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    e = ev.d["test"]
    rec = ev.recomendar(modelo, "test", [par])[par]
    rng = np.random.default_rng(SEMILLA)
    elegidos = rng.choice(e["con"], n, replace=False)
    nombres = dict(zip(ev.cat["item_id"], ev.cat["nombre"]))
    fig, axs = plt.subplots(n, n_compras + n_rec, figsize=(2 * (n_compras + n_rec), 2.6 * n))
    for r, c in enumerate(elegidos):
        compras = ev.tx[(ev.tx["customer_id"] == c) & (ev.tx["day"] < dia(CORTE["test"]))]
        compras = compras.sort_values("day")["item_id"].tolist()[-n_compras:]
        celdas = [(i, "gray") for i in compras] + [
            (i, "green" if i in e["reales"][c] else "red") for i in rec[c][:n_rec]]
        for k in range(n_compras + n_rec):
            ax = axs[r, k]; ax.set_xticks([]); ax.set_yticks([])
            if k < len(celdas):
                i, color = celdas[k]
                ruta = f"data/raw/imagenes/{i}.jpg"
                if os.path.exists(ruta):
                    ax.imshow(Image.open(ruta))
                ax.set_title(nombres[i][:22], fontsize=7)
                for s in ax.spines.values():
                    s.set_edgecolor(color); s.set_linewidth(4 if color == "green" else 1.5)
            else:
                ax.axis("off")
        axs[r, 0].set_ylabel(f"cliente {c[:6]}", fontsize=8)
    fig.suptitle("Últimas 5 compras (gris) y 6 recomendaciones (verde = comprado en test)")
    fig.tight_layout()
    fig.savefig("reports/figuras/ejemplos_B.png", dpi=120)
    plt.close(fig)


def corrida(nombre, cfg, opc, data, ev, tamanos, epocas, version):
    """Entrena una config y elige (alpha, beta) en val. Devuelve (fila para iteraciones.csv, modelo, info, par)."""
    print(f"\n{nombre}: d={cfg[0]} lr={cfg[1]} batch={cfg[2]} {opc}")
    modelo, info = entrenar(cfg, opc, data, ev, tamanos, epocas)
    par, val = elegir(modelo, ev)
    a0 = max(ALFAS, key=lambda x: val[(x, 0)][f"recall@{K}"])  # sin recompra
    test = ev.metricas(modelo, "test", list({(0, 0), (a0, 0), par}))
    r = lambda m, p: m[p][f"recall@{K}"]
    print(f"  (α, β) = {par}: val {r(val, par):.4f}, test {r(test, par):.4f} | sin recompra α={a0}: "
          f"val {r(val, (a0, 0)):.4f}, test {r(test, (a0, 0)):.4f} | puro: test {r(test, (0, 0)):.4f} | "
          f"{info['segundos']:.0f} s", flush=True)
    fila_ = {"version": version, "iteracion": nombre, "d": cfg[0], "lr": cfg[1], "batch": cfg[2],
             "epocas": info["epocas"], "mejor_epoca": info["mejor_epoca"], "alpha": par[0], "beta": par[1],
             "val_recall@12": r(val, par), "val_precision@12": val[par][f"precision@{K}"],
             "val_map@12": val[par][f"map@{K}"],
             "test_recall@12": r(test, par), "test_precision@12": test[par][f"precision@{K}"],
             "test_map@12": test[par][f"map@{K}"],
             "val_recall_sin_recompra": r(val, (a0, 0)), "test_recall_sin_recompra": r(test, (a0, 0)),
             "val_recall_alpha0": r(val, (0, 0)), "test_recall_alpha0": r(test, (0, 0)),
             "segundos": round(info["segundos"], 1), "device": device}
    return fila_, modelo, info, par


def ablacion(ev, modelo, version, par):
    """Filas de ablación: A, pop + recompra (sin modelo), y {puro, + pop, + pop + recompra} de `modelo`."""
    def f(variante, ver, m, p):
        v, t = m[0][p], m[1][p]
        return {"variante": variante, "version": ver, "alpha": p[0], "beta": p[1],
                **{f"val_{k}@12": v[f"{k}@{K}"] for k in ["recall", "precision", "map"]},
                **{f"test_{k}@12": t[f"{k}@{K}"] for k in ["recall", "precision", "map"]}}
    ab = pd.read_csv(VERSION_A)
    ab = ab[ab["grupo"] == "todos"].set_index("split")
    filas = [{"variante": "A: popularidad", "version": "A", "alpha": np.nan, "beta": np.nan,
              **{f"{s}_{k}@12": ab.loc[s, f"{k}@{K}"] for s in ["val", "test"] for k in ["recall", "precision", "map"]}}]
    pn = [p for p in PARES if p != (0, 0)]  # (0, 0) sin modelo es un empate total
    pv = ev.metricas(None, "val", pn)
    p0 = max(pn, key=lambda p: pv[p][f"recall@{K}"])
    filas.append(f("pop + recompra (sin modelo)", "-", (pv, ev.metricas(None, "test", [p0])), p0))
    a0 = elegir(modelo, ev, [(a, 0) for a in ALFAS])[0]
    for nombre, p in [(f"{version} puro", (0, 0)), (f"{version} + pop", a0), (f"{version} + pop + recompra", par)]:
        filas.append(f(nombre, version, (ev.metricas(modelo, "val", [p]), ev.metricas(modelo, "test", [p])), p))
    nuevo = pd.DataFrame(filas)
    if os.path.exists(ABLACION):
        previo = pd.read_csv(ABLACION)
        nuevo = pd.concat([previo[~previo["version"].isin([version, "-", "A"])], nuevo], ignore_index=True)
    nuevo = nuevo.sort_values("version", key=lambda c: c.map({"A": 0, "-": 1, "B": 2, "C": 3}), kind="stable")
    nuevo.to_csv(ABLACION, index=False, float_format="%.6f")
    return nuevo


class BaseCero:
    """Modelo base sin entrenar: item = embedding fijo normalizado; cliente = promedio por recencia."""

    def __init__(self, fijos):
        self.I = F.normalize(torch.from_numpy(fijos).to(device), dim=1)

    def eval(self):
        pass

    def items(self):
        return self.I

    def clientes(self, I, hist, dias):
        w = torch.exp(-dias / VIDA_DIAS) * (hist >= 0)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        return F.normalize((w.unsqueeze(-1) * I[hist.clamp(min=0)]).sum(1), dim=1)


def transfer(a, data, ev, tamanos, version):
    """Completo (checkpoint) vs base zero-shot vs two-tower sin preentrenados -> transfer_{version}.csv."""
    cat, tx, fijos, cats, vocab, clientes = data
    ck = torch.load(f"data/modelos/version_{version}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    cfg, opc, par_c = (c["d"], c["lr"], c["batch"]), c["opciones"], (c["alpha"], c["beta"])
    completo = TwoTower(fijos, cats, tamanos, c["d"], opc["id"]).to(device)
    completo.load_state_dict(ck["state_dict"])
    if not opc["id"]:
        print("AVISO: las opciones ganadoras tienen id=False; la variante sin preentrenados usa solo categóricas")
    print(f"\nSin preentrenados: d={cfg[0]} lr={cfg[1]} batch={cfg[2]} {opc}, fijos de ancho 0")
    sin, _ = entrenar(cfg, opc, (cat, tx, fijos[:, :0], cats, vocab, clientes), ev, tamanos, a.epocas)
    variantes = [("base sin entrenar (zero-shot)", BaseCero(fijos), None),
                 ("two-tower sin preentrenados", sin, None), ("modelo completo", completo, par_c)]
    if version == "C":  # texto + imagen con la config y opciones del checkpoint de B (no se guarda)
        cb = torch.load("data/modelos/version_B.pt", map_location="cpu", weights_only=False)["config"]
        cfg_b = (cb["d"], cb["lr"], cb["batch"])
        print(f"\nC con config de B: d={cfg_b[0]} lr={cfg_b[1]} batch={cfg_b[2]} {cb['opciones']}")
        con_b, info_b = entrenar(cfg_b, cb["opciones"], data, ev, tamanos, a.epocas)
        print(f"  épocas {info_b['epocas']}, mejor época {info_b['mejor_epoca']}")
        variantes.append(("con config de B", con_b, None))

    def f(variante, m, p):
        v, t = ev.metricas(m, "val", [p])[p], ev.metricas(m, "test", [p])[p]
        return {"variante": variante, "version": version, "alpha": p[0], "beta": p[1],
                **{f"val_{k}@12": v[f"{k}@{K}"] for k in ["recall", "precision", "map"]},
                **{f"test_{k}@12": t[f"{k}@{K}"] for k in ["recall", "precision", "map"]}}
    filas = []
    for nombre, m, par in variantes:
        par = par or elegir(m, ev)[0]
        print(f"  {nombre}: (alpha, beta) = {par}")
        filas += [f(f"{version} {nombre} puro", m, (0, 0)), f(f"{version} {nombre} + pop + recompra", m, par)]
    df = pd.DataFrame(filas)
    base = df.iloc[[0, 1] * (len(df) // 2)]["test_recall@12"].values  # puro vs puro, combinado vs combinado
    df["delta_test_recall_vs_base"] = df["test_recall@12"].values - base
    df.to_csv(f"reports/resultados/transfer_{version}.csv", index=False, float_format="%.6f")
    pd.options.display.float_format = "{:.4f}".format
    print(f"\nTransfer learning versión {version}:\n" + df.to_string(index=False))

    ref = pd.read_csv(f"reports/resultados/version_{version}.csv").query("split == 'test' and grupo == 'todos'")
    guardado = ref["recall@12"].iloc[0]
    assert abs(df.loc[5, "test_recall@12"] - guardado) < 1e-4, \
        f"no reproduce version_{version}.csv: {df.loc[5, 'test_recall@12']:.6f} vs {guardado:.6f}"
    assert len(df) == (8 if version == "C" else 6)
    assert df[[k for k in df.columns if "@12" in k]].stack().between(0, 1).all()
    print(f"\nModelo completo reproduce test recall@12 = {guardado:.6f}. Guardado: transfer_{version}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--epocas", type=int, default=MAX_EPOCAS)
    ap.add_argument("--una", action="store_true")
    a = ap.parse_args()
    version = "C" if a.imagen else "B"

    data = cargar(a.imagen)
    cat, tx, fijos, cats, vocab, clientes = data
    tamanos = [len(vocab[c]) for c in CATEGORICAS]
    ev = Evaluador(cat, tx, clientes)
    print(f"device: {device}, versión {version}, fijos: {fijos.shape[1]} dims")
    if a.transfer:
        return transfer(a, data, ev, tamanos, version)

    filas, mejor = [], None
    def correr(nombre, cfg, opc):
        nonlocal mejor
        fila_, modelo, info, par = corrida(nombre, cfg, opc, data, ev, tamanos, a.epocas, version)
        filas.append(fila_)
        if mejor is None or fila_["val_recall@12"] > mejor[0]:
            estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            mejor = (fila_["val_recall@12"], cfg, opc, par, info, estado)
        return fila_

    for nombre, opc in ITERS:
        correr(nombre, BASE, opc)
    mejor_it = max(range(len(ITERS)), key=lambda i: filas[i]["val_recall@12"])
    print(f"\nMejor iteración: {ITERS[mejor_it][0]}")
    for cfg in (REJILLA[:1] if a.una else REJILLA):
        correr("it5 rejilla", cfg, ITERS[mejor_it][1])

    # guardar la mejor corrida (se recarga el modelo, no se reentrena)
    _, (d, lr, batch), opc, par, info, estado = mejor
    modelo = TwoTower(fijos, cats, tamanos, d, opc["id"]).to(device)
    modelo.load_state_dict(estado)
    os.makedirs("data/modelos", exist_ok=True)
    torch.save({"state_dict": estado, "config": dict(d=d, lr=lr, batch=batch, alpha=par[0], beta=par[1],
                                                     imagen=a.imagen, opciones=opc, vocabularios=vocab)},
               f"data/modelos/version_{version}.pt")
    res = tabla_resultados(ev, modelo, par)
    res.to_csv(f"reports/resultados/version_{version}.csv", index=False, float_format="%.6f")
    it = pd.DataFrame(filas)
    if os.path.exists(ITERACIONES):
        previo = pd.read_csv(ITERACIONES)
        it = pd.concat([previo[previo["version"] != version], it], ignore_index=True)
    it.to_csv(ITERACIONES, index=False, float_format="%.6f")
    abl = ablacion(ev, modelo, version, par)
    curva(info, version, d, lr, batch)
    if version == "B":
        ejemplos(ev, modelo, par)

    pd.options.display.float_format = "{:.4f}".format
    print(f"\nMejor config: d={d} lr={lr} batch={batch} {opc} (alpha, beta)={par}")
    print(f"Métricas versión {version}:\n" + res.to_string(index=False))
    print("\nAblación:\n" + abl.to_string(index=False))

    assert res[["recall@12", "precision@12", "map@12"]].stack().between(0, 1).all()
    assert {("val", "todos"), ("test", "todos")} <= set(zip(res["split"], res["grupo"]))
    assert len(cat) == fijos.shape[0] == cats.shape[0]
    assert len(filas) == len(ITERS) + (1 if a.una else len(REJILLA))
    assert abl[[c for c in abl.columns if "@12" in c]].stack().between(0, 1).all()
    print(f"\nGuardado: reports/resultados/version_{version}.csv, ablacion.csv, data/modelos/version_{version}.pt")


if __name__ == "__main__":
    main()
