"""
Incertidumbre de las métricas: bootstrap sobre clientes de test y variación entre semillas.

Entrada:  data/modelos/version_{B,C}.pt, data/processed/*.csv, data/embeddings/*.npz,
          reports/resultados/ablacion.csv  (solo para verificar la reproducción)
Salida:   reports/resultados/incertidumbre.csv   recall@12 de test por sistema y diferencias pareadas
                                                 (en recall y en Q/mes) con IC 95 % por percentiles
          reports/resultados/semillas.csv        B, C con config de B y B sin preentrenados con 5 semillas
                                                 (+ filas de media y desviación estándar, ddof=1)

Parte 1 (sin reentrenar): recall@12 por cliente de test (todos, también sin historial) de A, la regla
  pop + recompra (alpha, beta elegidos en val, sin modelo), B y C; se remuestrean los clientes con
  reemplazo (BOOT veces, mismos clientes para todos los sistemas). El extra en Q/mes de cada remuestreo
  es modelo_valor.extra_neto(recall_sistema, recall_A, INSTANCIAS[sistema], base()).
Parte 2: reentrena con las semillas 42-46 usando config y opciones de version_B.pt (no se guardan modelos).

Uso:
  python src/evaluation/incertidumbre.py   (desde la raíz del repo; unos 15 min en mps)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.metricas import _recall, _top_k  # noqa: E402
from evaluation.modelo_valor import ABLACION, FILAS, INSTANCIAS, base, extra_neto  # noqa: E402
from models.baseline import K  # noqa: E402
import models.two_tower as tt  # noqa: E402

SALIDA_1 = "reports/resultados/incertidumbre.csv"
SALIDA_2 = "reports/resultados/semillas.csv"
BOOT, SEMILLA_BOOT, SEMILLAS = 2000, 42, [42, 43, 44, 45, 46]
METRICAS = ["val_puro", "val_comb", "test_puro", "test_comb"]


def por_cliente(rec, reales):
    """recall@12 de cada cliente de test (mismo orden y definición que metricas.recall_at_k)."""
    return np.array([_recall(_top_k(rec.get(c, []), K), r, K) for c, r in reales.items() if r])


def cargar_modelo(version, data, tamanos):
    ck = torch.load(f"data/modelos/version_{version}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    m = tt.TwoTower(data[2], data[3], tamanos, c["d"], c["opciones"]["id"]).to(tt.device)
    m.load_state_dict(ck["state_dict"])
    return m, (c["alpha"], c["beta"])


def ic(x):
    return np.percentile(x, [2.5, 97.5])


def parte1(ev, dato_b, dato_c, tamanos):
    reales = ev.d["test"]["reales"]
    modelos = {"B": cargar_modelo("B", dato_b, tamanos), "C": cargar_modelo("C", dato_c, tamanos)}
    pn = [p for p in tt.PARES if p != (0, 0)]  # igual que ablacion con modelo=None
    p_regla = tt.elegir(None, ev, pn)[0]
    print(f"regla sin modelo: (alpha, beta) = {p_regla}")
    r = {"A": por_cliente(ev.d["test"]["pop"], reales),
         "Regla sin modelo": por_cliente(ev.recomendar(None, "test", [p_regla])[p_regla], reales)}
    for v, (m, p) in modelos.items():
        r[v] = por_cliente(ev.recomendar(m, "test", [p])[p], reales)

    ab = pd.read_csv(ABLACION).set_index("variante")
    for v, x in r.items():
        guardado = ab.loc[FILAS[v], "test_recall@12"]
        assert len(x) == len(reales) and abs(x.mean() - guardado) < 1e-4, f"{v}: {x.mean():.6f} vs {guardado:.6f}"
        print(f"{v}: promedio por cliente {x.mean():.6f}, guardado {guardado:.6f}")

    n = len(r["A"])
    rng = np.random.default_rng(SEMILLA_BOOT)
    boot = {v: np.empty(BOOT) for v in r}
    for b in range(BOOT):
        idx = rng.integers(0, n, n)
        for v, x in r.items():
            boot[v][b] = x[idx].mean()
    s = base()
    def q(v, rec, rec_a):  # Q/mes del sistema v respecto a A (función de modelo_valor)
        return extra_neto(rec, rec_a, INSTANCIAS[v], s)
    q_pt = {v: q(v, r[v].mean(), r["A"].mean()) for v in r}
    q_boot = {v: q(v, boot[v], boot["A"]) for v in r}

    filas = []
    for v in r:
        lo, hi = ic(boot[v])
        filas.append(dict(tipo="recall", nombre=v, estimado=r[v].mean(), ic_inf=lo, ic_sup=hi, frac_pos=np.nan))
    for tipo, pt, bt in [("dif_recall", {v: r[v].mean() for v in r}, boot), ("dif_Q_mes", q_pt, q_boot)]:
        pares = [("B", "Regla sin modelo"), ("B", "A"), ("Regla sin modelo", "A")]
        if tipo == "dif_recall":
            pares += [("C", "B"), ("C", "Regla sin modelo")]
        for a, b in pares:
            d = bt[a] - bt[b]
            lo, hi = ic(d)
            filas.append(dict(tipo=tipo, nombre=f"{a} - {b}", estimado=pt[a] - pt[b], ic_inf=lo, ic_sup=hi,
                              frac_pos=(d > 0).mean()))
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA_1, index=False, float_format="%.6f")
    print(df.to_string(index=False))
    return df


def parte2(data_b, data_c, ev, tamanos):
    cat, tx, fijos, cats, vocab, clientes = data_b
    cb = torch.load("data/modelos/version_B.pt", map_location="cpu", weights_only=False)["config"]
    cfg, opc = (cb["d"], cb["lr"], cb["batch"]), cb["opciones"]
    variantes = [("B (texto)", data_b), ("C con config de B (texto + imagen)", data_c),
                 ("B sin preentrenados", (cat, tx, fijos[:, :0], cats, vocab, clientes))]
    filas = []
    for nombre, data in variantes:
        for sem in SEMILLAS:
            print(f"\n{nombre}, semilla {sem}: d={cfg[0]} lr={cfg[1]} batch={cfg[2]} {opc}", flush=True)
            modelo, info = tt.entrenar(cfg, opc, data, ev, tamanos, tt.MAX_EPOCAS, sem)
            par, val = tt.elegir(modelo, ev)
            test = ev.metricas(modelo, "test", [(0, 0), par])
            f = lambda m, p: m[p][f"recall@{K}"]
            filas.append(dict(variante=nombre, semilla=sem, epocas=info["epocas"], mejor_epoca=info["mejor_epoca"],
                              alpha=par[0], beta=par[1], val_puro=f(val, (0, 0)), val_comb=f(val, par),
                              test_puro=f(test, (0, 0)), test_comb=f(test, par)))
            print(f"  (alpha, beta) = {par}: " + ", ".join(f"{k} {filas[-1][k]:.6f}" for k in METRICAS), flush=True)
    corridas = pd.DataFrame(filas)
    cols = ["epocas", "mejor_epoca"] + METRICAS
    res = []
    for nombre, g in corridas.groupby("variante", sort=False):
        res.append(dict(variante=nombre, semilla="media", **g[cols].mean().to_dict()))
        res.append(dict(variante=nombre, semilla="desv_std", **g[cols].std(ddof=1).to_dict()))
    df = pd.concat([corridas, pd.DataFrame(res)], ignore_index=True)
    df.to_csv(SALIDA_2, index=False, float_format="%.6f")
    pd.options.display.float_format = "{:.6f}".format
    print("\n" + df.to_string(index=False))
    return corridas, df


def main():
    data_b, data_c = tt.cargar(False), tt.cargar(True)
    cat, tx, fijos, cats, vocab, clientes = data_b
    tamanos = [len(vocab[c]) for c in tt.CATEGORICAS]
    ev = tt.Evaluador(cat, tx, clientes)
    print(f"device: {tt.device}")
    df1 = parte1(ev, data_b, data_c, tamanos)
    corridas, df2 = parte2(data_b, data_c, ev, tamanos)

    assert df1[["estimado", "ic_inf", "ic_sup"]][df1["tipo"] == "recall"].stack().between(0, 1).all()
    assert len(df1) == 4 + 5 + 3 and len(corridas) == 15 and len(df2) == 15 + 6
    assert corridas[METRICAS].stack().between(0, 1).all()
    print(f"\nGuardado: {SALIDA_1}, {SALIDA_2}")


if __name__ == "__main__":
    main()
