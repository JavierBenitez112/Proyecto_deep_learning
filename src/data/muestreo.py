"""
Muestra de las transacciones de H&M (últimos 6 meses, 50,000 clientes).

Entrada:  data/raw/hm/transactions_train.csv  (3.2 GB, se lee por chunks)
Salida:   data/processed/transacciones_hm.csv
            t_dat, customer_id, article_id, price, sales_channel_id, split

Pasos:
  1. Se queda con las compras de la ventana INICIO..FIN (el dataset termina el 2020-09-22).
  2. Muestrea N_CLIENTES al azar entre los que compraron en la ventana (semilla fija).
  3. Split temporal: test = última semana, val = la semana anterior, train = el resto.

Uso:
  python src/data/muestreo.py   (desde la raíz del repo)
"""

import os
import numpy as np
import pandas as pd

ENTRADA = "data/raw/hm/transactions_train.csv"
SALIDA = "data/processed/transacciones_hm.csv"

FILAS_CHUNK = 5_000_000
INICIO = "2020-03-24"       # 6 meses antes del fin del dataset
INICIO_VAL = "2020-09-09"   # val: 09-09 a 09-15
INICIO_TEST = "2020-09-16"  # test: 09-16 a 09-22
FIN = "2020-09-22"
N_CLIENTES = 50_000
SEMILLA = 42

# article_id va como texto: tiene ceros a la izquierda. Las fechas ISO se comparan como texto.
DTYPES = {"t_dat": str, "customer_id": str, "article_id": str,
          "price": "float64", "sales_channel_id": "int8"}


def leer_ventana():
    """Lee el CSV por chunks y junta solo las filas de la ventana INICIO..FIN."""
    partes, total = [], 0
    for chunk in pd.read_csv(ENTRADA, dtype=DTYPES, chunksize=FILAS_CHUNK):
        total += len(chunk)
        partes.append(chunk[(chunk["t_dat"] >= INICIO) & (chunk["t_dat"] <= FIN)])
        print(f"  leídas {total:>11,} filas   en ventana: {sum(map(len, partes)):>10,}", flush=True)
    return pd.concat(partes, ignore_index=True), total


def asignar_split(fechas):
    """test / val / train según la fecha (texto ISO)."""
    return np.select([fechas >= INICIO_TEST, fechas >= INICIO_VAL], ["test", "val"], "train")


def main():
    ventana, total = leer_ventana()
    print(f"\nFilas totales: {total:,}   en la ventana {INICIO}..{FIN}: {len(ventana):,}")
    assert ventana["t_dat"].max() == FIN, "el dataset debería terminar el " + FIN

    clientes = np.sort(ventana["customer_id"].unique())
    rng = np.random.default_rng(SEMILLA)
    elegidos = rng.choice(clientes, N_CLIENTES, replace=False)
    print(f"Clientes que compraron en la ventana: {len(clientes):,}   muestreados: {len(elegidos):,}")

    df = ventana[ventana["customer_id"].isin(elegidos)].copy()
    df["split"] = asignar_split(df["t_dat"].to_numpy())
    df = df.sort_values("t_dat", kind="stable", ignore_index=True)

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)

    resumen = df.groupby("split").agg(filas=("article_id", "size"),
                                      clientes=("customer_id", "nunique"),
                                      articulos=("article_id", "nunique"))
    print("\n" + resumen.loc[["train", "val", "test"]].to_string())
    en_train = set(df.loc[df["split"] == "train", "customer_id"])
    for s in ("val", "test"):
        cs = set(df.loc[df["split"] == s, "customer_id"])
        print(f"Clientes de {s} con historial en train: {len(cs & en_train):,} de {len(cs):,} "
              f"({len(cs & en_train) / len(cs):.1%})")

    assert df["customer_id"].nunique() <= N_CLIENTES and (resumen["filas"] > 0).all()
    assert df.loc[df["split"] == "train", "t_dat"].max() < INICIO_VAL
    assert df["t_dat"].min() >= INICIO and df["t_dat"].max() == FIN
    assert df["article_id"].str.len().eq(10).all()  # no se perdieron los ceros
    print(f"\nGuardado en {SALIDA}  ({len(df):,} filas)")


if __name__ == "__main__":
    main()
