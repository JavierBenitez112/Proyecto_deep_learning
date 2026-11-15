"""
Recomendador reutilizable (versiones A, B, C + cold start por imagen) para la API y la web.

Entrada:  data/processed/{catalogo,transacciones}.csv, data/embeddings/{texto,imagen}.npz,
          data/modelos/version_{B,C}.pt
Salida:   listas de dicts {item_id, nombre, marca, precio_q, imagen_url, link, puntaje, razon}
Momento de la demo: historial = TODAS las transacciones (hasta 2020-09-22);
          popularidad = última semana (2020-09-16 a 2020-09-22).

Regla de la razón (B y C), en este orden:
  1. el item está en el historial del cliente  -> "Lo compraste hace N días"
  2. alpha * z(popularidad) > cos(cliente, item) -> "Popular esta semana"
  3. si no                                     -> "Parecido a «X», que compraste"
     (X = prenda del historial con mayor coseno con el item en el espacio del modelo)
  A: "Top ventas de la semana". Cliente nuevo/sin historial: popularidad, como A.

Uso:
  python src/models/recommend.py     (desde la raíz; corre los asserts)
  from models.recommend import Recomendador; r = Recomendador(); r.recomendar(cid, "B")
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.two_tower import cargar, historial, TwoTower, dia, CATEGORICAS, VIDA_DIAS, DIM_CAT, IMAGEN  # noqa: E402
from models.baseline import mas_comprados  # noqa: E402

HOY = pd.Timestamp("2020-09-23")          # día "actual" de la demo
SEMANA = ("2020-09-16", "2020-09-22")
NUEVO = "Cliente nuevo: lo más vendido esta semana"
torch.set_num_threads(2)


class Recomendador:
    def __init__(self):
        torch.set_grad_enabled(False)
        self.cat, self.tx, _, _, _, self.clientes = cargar(False)
        self.cod = {c: i for i, c in enumerate(self.clientes)}
        self.ids = self.cat["item_id"].values
        self.pos = {v: i for i, v in enumerate(self.ids)}
        self.info = self.cat[["item_id", "nombre", "marca", "precio_q", "imagen_url", "link"]].fillna("").to_dict("records")
        corte = dia(HOY)
        self.hist, self.dias = historial(self.tx, len(self.clientes), corte)
        self.corte = corte
        self.top_pop = mas_comprados(self.tx, *SEMANA)
        # z(log(1 + compras de la semana)), igual que en el entrenamiento
        sem = self.tx[(self.tx["t_dat"] >= SEMANA[0]) & (self.tx["t_dat"] <= SEMANA[1])]
        l = np.log1p(np.bincount(sem["ii"].values, minlength=len(self.cat)))
        self.z = torch.tensor((l - l.mean()) / l.std(), dtype=torch.float32)
        # modelos en CPU: embeddings de items precalculados una vez
        self.m = {}
        for v, imagen in [("B", False), ("C", True)]:
            ck = torch.load(f"data/modelos/version_{v}.pt", map_location="cpu", weights_only=False)
            c = ck["config"]
            _, _, fijos, cats, _, _ = cargar(imagen)
            tam = [len(c["vocabularios"][k]) for k in CATEGORICAS]
            modelo = TwoTower(fijos, cats, tam, c["d"], c["opciones"]["id"])
            modelo.load_state_dict(ck["state_dict"])
            modelo.eval()
            self.m[v] = dict(modelo=modelo, I=modelo.items(), alpha=c["alpha"], beta=c["beta"])
        zi = np.load(IMAGEN, allow_pickle=True)
        self.img = torch.from_numpy(zi["emb"]).float()
        self.tiene = torch.from_numpy(zi["tiene_imagen"].astype(bool))
        self.clip = None  # perezoso: solo cuando llega una foto

    # ---------- utilidades ----------

    def _item(self, j, puntaje, razon):
        return {**self.info[j], "puntaje": round(float(puntaje), 4), "razon": razon}

    def _populares(self, k, razon):
        return [self._item(self.pos[i], 0, razon) for i in self.top_pop[:k]]

    def _fila(self, cliente_id):
        """Fila del cliente si tiene historial; si no, None."""
        f = self.cod.get(cliente_id)
        return f if f is not None and (self.hist[f] >= 0).any() else None

    # ---------- API pública ----------

    def recomendar(self, cliente_id, version="B", k=12):
        """Top-k para un cliente. version: 'A' (más vendidos), 'B' (two-tower), 'C' (multimodal)."""
        f = self._fila(cliente_id)
        if version == "A":
            return self._populares(k, "Top ventas de la semana")
        if f is None:
            return self._populares(k, NUEVO)
        m = self.m[version]
        I, a = m["I"], m["alpha"]
        h = torch.from_numpy(self.hist[f:f + 1])
        d = torch.from_numpy(self.dias[f:f + 1])
        u = m["modelo"].clientes(I, h, d)[0]
        w = torch.exp(-d[0] / VIDA_DIAS) * (h[0] >= 0)
        r = torch.zeros(len(self.z)).scatter_add_(0, h[0].clamp(min=0), w)  # recompra
        cos = I @ u
        top = (cos + a * self.z + m["beta"] * r).topk(k)
        # historial completo del cliente (item -> días desde la última compra) para las razones
        compras = self.tx[self.tx["cc"] == f]
        ultima = (self.corte - compras.groupby("ii")["day"].max()).to_dict()
        prev = torch.tensor(list(ultima), dtype=torch.long)
        out = []
        for s, j in zip(top.values.tolist(), top.indices.tolist()):
            if j in ultima:
                razon = f"Lo compraste hace {ultima[j]} días"
            elif a * self.z[j] > cos[j]:
                razon = "Popular esta semana"
            else:
                p = prev[(I[prev] @ I[j]).argmax()].item()
                razon = f"Parecido a «{self.info[p]['nombre']}», que compraste"
            out.append(self._item(j, s, razon))
        return out

    def historial(self, cliente_id, n=6):
        """Últimas n compras del cliente (más reciente primero) con fecha."""
        f = self.cod.get(cliente_id)
        if f is None:
            return []
        c = self.tx[self.tx["cc"] == f].sort_values("day", kind="stable").tail(n).iloc[::-1]
        return [{**self.info[j], "fecha": t} for j, t in zip(c["ii"], c["t_dat"])]

    def similares(self, item_id=None, foto=None, k=12):
        """Cold start por coseno de imagen CLIP. foto = bytes o ruta; item_id = prenda del catálogo (se excluye)."""
        if foto is not None:
            if self.clip is None:
                from transformers import CLIPModel, CLIPProcessor
                from models.embeddings import MODELO_IMG
                self.clip = (CLIPModel.from_pretrained(MODELO_IMG).eval(), CLIPProcessor.from_pretrained(MODELO_IMG))
            from models.embeddings import emb_foto
            ruta = io.BytesIO(foto) if isinstance(foto, bytes) else foto
            q, excluir = torch.from_numpy(emb_foto(ruta, *self.clip, "cpu")).float(), -1
        else:
            excluir = self.pos[str(item_id)]
            q = self.img[excluir]
        s = (self.img @ q).masked_fill(~self.tiene, -2)
        if excluir >= 0:
            s[excluir] = -2
        top = s.topk(k)
        return [self._item(j, v, f"Similitud visual {v:.2f}") for v, j in zip(top.values.tolist(), top.indices.tolist())]

    def clientes_demo(self, n=20):
        """n clientes con historial y compras en test (semilla 42): [{cliente_id, compras}]."""
        test = set(self.tx[self.tx["split"] == "test"]["customer_id"])
        ok = [c for c in sorted(test) if c in self.cod and (self.hist[self.cod[c]] >= 0).any()]
        cuenta = self.tx.groupby("cc").size()
        pick = np.random.RandomState(42).choice(len(ok), n, replace=False)
        return [{"cliente_id": ok[i], "compras": int(cuenta[self.cod[ok[i]]])} for i in pick]


if __name__ == "__main__":
    r = Recomendador()
    cid = r.clientes_demo(1)[0]["cliente_id"]
    for v in "ABC":
        rec = r.recomendar(cid, v)
        assert len({x["item_id"] for x in rec}) == 12, v
        print(v, [x["razon"] for x in rec[:3]])
    nuevo = r.recomendar("no-existe", "C")
    assert [x["item_id"] for x in nuevo] == [x["item_id"] for x in r.recomendar(cid, "A")]
    assert nuevo[0]["razon"] == NUEVO
    sim = r.similares(item_id=r.ids[0])
    assert len(sim) == 12 and r.ids[0] not in [x["item_id"] for x in sim]
    i = int(np.flatnonzero(r.tiene.numpy())[5])
    foto = r.similares(foto=f"data/raw/imagenes/{r.ids[i]}.jpg")
    assert foto[0]["item_id"] == r.ids[i], foto[0]
    print("ok", foto[0]["razon"])
