"""
Modelo de valor (Fase 3): traduce recall@12 a quetzales al mes para las versiones A, B y C.

Entrada:  reports/resultados/ablacion.csv   (recall@12 de test de cada variante)
          data/processed/transacciones.csv  (solo para recall@k de A y de la regla sin modelo)
Salida:   reports/resultados/valor.csv         una fila por versión
          reports/resultados/sensibilidad.csv  cada supuesto +-20 % y +-50 %, y escenarios
          reports/figuras/valor_curva_recall.png, valor_curva_k.png, valor_sensibilidad.png

Fórmula (por versión v, A es la referencia):
  ventas_v = visitas x f_ve x CTR_v x conv_click x ticket
  CTR_v    = ctr_ref x recall_v / recall_A          (el CTR crece proporcional al recall)
  costo_v  = horas_mes x precio_hora_usd x tc x instancias_v
  extra_v  = (ventas_v - ventas_A) - (costo_v - costo_A)     [Q al mes]
donde
  visitas     visitas mensuales al sitio
  f_ve        fracción de visitas que ven el carrusel de recomendaciones
  ctr_ref     CTR del carrusel con la lista de A (popularidad)
  recall_v    recall@12 de test (ablacion.csv); recall_v / recall_A escala el CTR
  conv_click  fracción de visitas que hicieron clic en el carrusel y terminan comprando
  ticket      Q por compra (3.55 prendas x precio medio, de transacciones.csv)
  instancias  A y la regla: 0 (sin modelo); B: 1 servidor CPU; C: 2 (CLIP => más RAM)
Cada supuesto con su fuente está en docs/supuestos_valor.md.

Alcance: el extra mide ventas atribuidas al carrusel que A no captura; supone que
mostrar más aciertos sube el CTR de forma proporcional (aproximación lineal).

Uso:
  python src/evaluation/modelo_valor.py   (desde la raíz del repo)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))

ABLACION = "reports/resultados/ablacion.csv"
VALOR = "reports/resultados/valor.csv"
SENSIBILIDAD = "reports/resultados/sensibilidad.csv"
FIGURAS = "reports/figuras"
HORAS_MES = 730  # 24 h x 365 / 12

# supuesto: (valor, bajo, alto, descripción, fuente)   bajo/alto = rango plausible (escenarios)
SUPUESTOS = {
    "visitas":    (321_000, 226_000, 600_000, "visitas/mes al sitio", "Semrush (321K sep-2026; 226K feb-2026, solo tráfico El Salvador del dominio)"),
    "f_ve":       (0.50, 0.25, 0.75, "fracción de visitas que ven el carrusel", "supuesto del equipo"),
    "ctr_ref":    (0.02, 0.01, 0.04, "CTR del carrusel con la lista de A", "supuesto del equipo (blogs/tests: 1.4 %-6.5 %)"),
    "conv_click": (0.036, 0.008, 0.036, "compra tras clic en el carrusel", "Salesforce: 0.8 % LatAm x 4.5 (clic en recomendación)"),
    "ticket":     (896.0, 400.0, 1100.0, "Q por compra", "datos del proyecto (transacciones.csv); Cámara de Comercio 2020: Q170"),
    "tc":         (7.63, 7.50, 7.80, "Q por USD", "Banguat vía prensa, abr-2026"),
    "precio_hora_usd": (0.03, 0.03, 0.06, "USD/h CPU Upgrade", "Hugging Face Spaces (doc oficial)"),
}
INSTANCIAS = {"A": 0, "Regla sin modelo": 0, "B": 1, "C": 2}
FILAS = {"A": "A: popularidad", "Regla sin modelo": "pop + recompra (sin modelo)",
         "B": "B + pop + recompra", "C": "C + pop + recompra"}


def base():
    return {k: v[0] for k, v in SUPUESTOS.items()}


def ventas(recall, recall_a, s):
    """Ventas/mes (Q) atribuidas al carrusel con recall@12 = recall."""
    return s["visitas"] * s["f_ve"] * s["ctr_ref"] * (recall / recall_a) * s["conv_click"] * s["ticket"]


def costo(instancias, s):
    """Costo de servidor (Q/mes)."""
    return HORAS_MES * s["precio_hora_usd"] * s["tc"] * instancias


def extra_neto(recall, recall_a, instancias, s):
    """(ventas - ventas_A) - costo, en Q/mes. A tiene instancias=0 y recall=recall_a => 0."""
    return ventas(recall, recall_a, s) - ventas(recall_a, recall_a, s) - costo(instancias, s)


def recalls_test():
    ab = pd.read_csv(ABLACION).set_index("variante")
    return {v: float(ab.loc[f, "test_recall@12"]) for v, f in FILAS.items()}


def tabla_valor(s=None):
    s = s or base()
    r = recalls_test()
    filas = []
    for v, rec in r.items():
        vta = ventas(rec, r["A"], s)
        c = costo(INSTANCIAS[v], s)
        filas.append({"version": v, "recall@12_test": rec, "ventas_mes_Q": vta,
                      "extra_ventas_vs_A_Q": vta - ventas(r["A"], r["A"], s),
                      "costo_servidor_Q": c, "extra_neto_Q": extra_neto(rec, r["A"], INSTANCIAS[v], s)})
    return pd.DataFrame(filas)


def sensibilidad():
    """Cada supuesto +-20 % y +-50 % (uno a la vez) y escenarios; impacto en el extra neto de B."""
    r = recalls_test()
    f = lambda s: extra_neto(r["B"], r["A"], INSTANCIAS["B"], s)
    b = base()
    e0 = f(b)
    filas = [{"supuesto": "base", "cambio": "base", "valor": np.nan, "extra_neto_B_Q": e0}]
    for k in SUPUESTOS:
        for p in (-0.5, -0.2, 0.2, 0.5):
            s = dict(b, **{k: b[k] * (1 + p)})
            filas.append({"supuesto": k, "cambio": f"{p:+.0%}", "valor": s[k], "extra_neto_B_Q": f(s)})
    pes = {k: (v[2] if k == "precio_hora_usd" else v[1]) for k, v in SUPUESTOS.items()}
    opt = {k: (v[1] if k == "precio_hora_usd" else v[2]) for k, v in SUPUESTOS.items()}
    filas += [{"supuesto": "todos", "cambio": "pesimista", "valor": np.nan, "extra_neto_B_Q": f(pes)},
              {"supuesto": "todos", "cambio": "optimista", "valor": np.nan, "extra_neto_B_Q": f(opt)}]
    df = pd.DataFrame(filas)
    df["delta_vs_base_Q"] = df["extra_neto_B_Q"] - e0
    df["delta_vs_base_pct"] = 100 * df["delta_vs_base_Q"] / e0
    return df


def recall_por_k(ks=(4, 8, 12, 24)):
    """recall@k de test de A y de la regla sin modelo (pop + recompra), sin tocar los módulos de models/.
    ponytail: B/C necesitan el modelo entrenado y top_items fija K=12; no se calculan aquí."""
    from evaluation.metricas import recall_at_k
    import models.two_tower as tt
    from models.baseline import mas_comprados, SEMANA_PREVIA
    kmax = max(ks)
    tt.K = kmax  # top_items lee K global en cada llamada
    cat, tx, _, _, _, clientes = tt.cargar(False)
    ev = tt.Evaluador(cat, tx, clientes)
    e = ev.d["test"]
    pop = mas_comprados(tx, *SEMANA_PREVIA["test"], k=kmax)
    top = tt.top_items(None, e["hist"], e["dias"], e["z"], [(0.05, 0.5)])[(0.05, 0.5)]
    rec_regla = {c: pop for c in e["reales"]}
    rec_regla.update({c: list(ev.ids[top[i]]) for i, c in enumerate(e["con"])})
    rec_a = {c: pop for c in e["reales"]}
    return pd.DataFrame([{"k": k, "A": recall_at_k(rec_a, e["reales"], k),
                          "Regla sin modelo": recall_at_k(rec_regla, e["reales"], k)} for k in ks])


# ---------- figuras ----------

COL = {"A": "#7f7f7f", "Regla sin modelo": "#e69f00", "B": "#0072b2", "C": "#009e73"}


def fig_curva_recall(val, ruta=f"{FIGURAS}/valor_curva_recall.png"):
    r = recalls_test()
    x = np.linspace(0, 0.2, 100)
    s = base()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [extra_neto(v, r["A"], 1, s) for v in x], color="#444", label="curva (1 servidor)")
    for _, f in val.iterrows():
        ax.scatter(f["recall@12_test"], f["extra_neto_Q"], s=70, color=COL[f["version"]], zorder=3,
                   label=f"{f['version']}: Q{f['extra_neto_Q']:,.0f}")
    ax.axhline(0, color="k", lw=.5)
    ax.set(xlabel="recall@12 (test)", ylabel="Extra neto vs A (Q/mes)", title="Valor mensual según recall@12")
    ax.legend(); fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)


def fig_curva_k(rk, ruta=f"{FIGURAS}/valor_curva_k.png"):
    r, s = recalls_test(), base()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for v in ["A", "Regla sin modelo"]:
        ax[0].plot(rk["k"], rk[v], "o-", color=COL[v], label=v)
        ax[1].plot(rk["k"], [extra_neto(x, r["A"], INSTANCIAS[v], s) for x in rk[v]], "o-", color=COL[v], label=v)
    ax[0].set(xlabel="k (recomendaciones mostradas)", ylabel="recall@k (test)", title="recall@k")
    ax[1].set(xlabel="k (recomendaciones mostradas)", ylabel="Extra neto vs A@12 (Q/mes)", title="Valor (CTR proporcional al recall)")
    for a in ax: a.set_xticks(rk["k"]); a.legend()
    fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)


def fig_tornado(sens, ruta=f"{FIGURAS}/valor_sensibilidad.png"):
    e0 = sens.loc[sens["supuesto"] == "base", "extra_neto_B_Q"].iloc[0]
    d = sens[~sens["supuesto"].isin(["base", "todos"])]
    ord_ = d.groupby("supuesto")["delta_vs_base_Q"].apply(lambda x: x.abs().max()).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(ord_):
        for p, c in [("-50%", "#d55e00"), ("-20%", "#e69f00"), ("+20%", "#56b4e9"), ("+50%", "#0072b2")]:
            ax.barh(i, d[(d.supuesto == k) & (d.cambio == p)]["delta_vs_base_Q"].iloc[0], color=c,
                    label=p if i == 0 else None, height=.6 if "50" in p else .35)
    ax.set_yticks(range(len(ord_)), ord_); ax.axvline(0, color="k", lw=.7)
    ax.set(xlabel="Cambio en el extra neto de B (Q/mes)", title=f"Sensibilidad (base: Q{e0:,.0f}/mes)")
    ax.legend(); fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)


def frases(val):
    t = val.set_index("version")
    return [f"{v} genera Q{t.loc[v, 'extra_ventas_vs_A_Q']:,.0f} más al mes que A, con un costo extra de Q{t.loc[v, 'costo_servidor_Q']:,.0f}"
            f" (extra neto Q{t.loc[v, 'extra_neto_Q']:,.0f})" for v in ["B", "C", "Regla sin modelo"]]


def main():
    os.makedirs(FIGURAS, exist_ok=True)
    val = tabla_valor(); val.to_csv(VALOR, index=False, float_format="%.4f")
    sens = sensibilidad(); sens.to_csv(SENSIBILIDAD, index=False, float_format="%.4f")
    rk = recall_por_k()
    fig_curva_recall(val); fig_curva_k(rk); fig_tornado(sens)
    print(val.to_string(index=False)); print(rk.to_string(index=False))
    print("\n".join(frases(val)))
    t = val.set_index("version")
    assert np.isfinite(val.select_dtypes("number").values).all() and np.isfinite(sens["extra_neto_B_Q"]).all()
    assert abs(t.loc["A", "extra_neto_Q"]) < 1e-9
    r = recalls_test(); s = base()
    for a, b in [("A", "B"), ("B", "C")]:  # a mismo costo, más recall => más extra
        for x, y in [(r[a], r[b])]:
            assert (extra_neto(x, r["A"], 1, s) < extra_neto(y, r["A"], 1, s)) == (x < y)
    assert extra_neto(r["Regla sin modelo"], r["A"], 1, s) < extra_neto(r["B"], r["A"], 1, s)
    print("ok")
    return val, sens, rk


if __name__ == "__main__":
    main()
