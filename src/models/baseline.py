"""
Versión A (línea base): recomendar a todos los clientes los 12 items más comprados.

Entrada:  data/processed/transacciones.csv, data/processed/catalogo.csv
Salida:   reports/resultados/version_A.csv   (recall@12, precision@12 y MAP@12)

Los 12 más comprados son los de la última semana antes del periodo evaluado:
  val  -> última semana de train (2020-09-02 a 2020-09-08)
  test -> semana de val          (2020-09-09 a 2020-09-15)
Los items reales de cada cliente son los que compró en el periodo evaluado.

Uso:
  python src/models/baseline.py   (desde la raíz del repo)
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.metricas import evaluar  # noqa: E402

TRANSACCIONES = "data/processed/transacciones.csv"
CATALOGO = "data/processed/catalogo.csv"
SALIDA = "reports/resultados/version_A.csv"

K = 12
# periodo evaluado -> semana de la que salen los más comprados
SEMANA_PREVIA = {"val": ("2020-09-02", "2020-09-08"), "test": ("2020-09-09", "2020-09-15")}


def mas_comprados(tx, inicio, fin, k=K):
    """Los k items con más compras entre inicio y fin (empates por item_id)."""
    semana = tx[(tx["t_dat"] >= inicio) & (tx["t_dat"] <= fin)]
    cuentas = semana["item_id"].value_counts().sort_index()
    return cuentas.sort_values(ascending=False, kind="stable").head(k).index.tolist()


def seccion_cliente(tx, cat):
    """Sección donde más compra cada cliente en train (empates: la primera en orden alfabético)."""
    train = tx[tx["split"] == "train"].merge(cat[["item_id", "seccion"]], on="item_id")
    cuentas = train.groupby(["customer_id", "seccion"]).size().reset_index(name="n")
    cuentas = cuentas.sort_values(["customer_id", "n", "seccion"], ascending=[True, False, True])
    return cuentas.drop_duplicates("customer_id").set_index("customer_id")["seccion"]


def fila(split, grupo, recomendados, reales):
    return {"split": split, "grupo": grupo, "clientes": len(reales),
            **evaluar(recomendados, reales, K)}


def main():
    cat = pd.read_csv(CATALOGO, dtype={"item_id": str})
    tx = pd.read_csv(TRANSACCIONES, dtype={"item_id": str, "customer_id": str})
    secciones = seccion_cliente(tx, cat)

    filas = []
    for split, (inicio, fin) in SEMANA_PREVIA.items():
        top = mas_comprados(tx, inicio, fin)
        print(f"\n{split}: top {K} de {inicio} a {fin}")
        for i, item in enumerate(top, 1):
            nombre = cat.loc[cat["item_id"] == item, "nombre"].iloc[0]
            print(f"  {i:>2}. {item}  {nombre}")

        reales = tx[tx["split"] == split].groupby("customer_id")["item_id"].agg(set).to_dict()
        recomendados = {c: top for c in reales}  # la misma lista para todos
        filas.append(fila(split, "todos", recomendados, reales))
        if split == "test":
            grupos = pd.Series({c: secciones.get(c, "sin historial") for c in reales})
            for s in sorted(grupos.unique()):
                reales_s = {c: reales[c] for c in grupos.index[grupos == s]}
                filas.append(fila(split, f"seccion={s}", recomendados, reales_s))

    res = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    res.to_csv(SALIDA, index=False, float_format="%.6f")

    pd.options.display.float_format = "{:.4f}".format
    print("\nMétricas versión A:\n" + res.to_string(index=False))

    assert res["recall@12"].between(0, 1).all() and res["map@12"].between(0, 1).all()
    assert res.loc[res["grupo"] == "todos", "clientes"].gt(0).all()
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
