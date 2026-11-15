"""
API FastAPI del recomendador (sirve también la web en "/").

Entrada:  HTTP (ver endpoints); --medir para medir latencia
Salida:   JSON {cliente_id, version, items, ms}; con --medir, reports/resultados/latencia.csv
Endpoints:
  GET  /recomendar/{cliente_id}?version=A|B|C&k=12
  POST /similares        multipart: foto (archivo) o item_id (form)
  GET  /historial/{cliente_id}     últimas 6 compras
  GET  /clientes         clientes demo
  GET  /salud
Costo por 1000 predicciones (SUPUESTO): servidor CPU de HF Spaces a 0.03 USD/h
  -> 0.03 / (req_s * 3600) * 1000.

Uso (desde la raíz del repo):
  .venv/bin/python -m uvicorn src.api.main:app --port 8000     y abrir http://localhost:8000
  .venv/bin/python src/api/main.py --medir
"""

import os
import sys
import time
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.recommend import Recomendador  # noqa: E402

USD_HORA = 0.03  # supuesto: CPU básica de HF Spaces
SALIDA = "reports/resultados/latencia.csv"
estado = {}


@asynccontextmanager
async def vida(app):
    estado["r"] = Recomendador()  # una sola vez al arrancar
    yield


app = FastAPI(title="Recomendador de moda", lifespan=vida)


@app.get("/")
def web():
    return FileResponse("web/index.html")


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/clientes")
def clientes():
    return estado["r"].clientes_demo()


@app.get("/historial/{cliente_id}")
def historial(cliente_id: str):
    return estado["r"].historial(cliente_id)


@app.get("/recomendar/{cliente_id}")
def recomendar(cliente_id: str, version: str = "B", k: int = 12):
    if version not in ("A", "B", "C"):
        raise HTTPException(400, "version debe ser A, B o C")
    t0 = time.perf_counter()
    items = estado["r"].recomendar(cliente_id, version, min(max(k, 1), 50))
    return {"cliente_id": cliente_id, "version": version, "items": items,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/similares")
async def similares(foto: UploadFile | None = File(None), item_id: str | None = Form(None)):
    if foto is None and not item_id:
        raise HTTPException(400, "envía foto o item_id")
    t0 = time.perf_counter()
    try:
        items = estado["r"].similares(item_id=item_id, foto=await foto.read() if foto else None)
    except Exception as e:  # item inexistente o archivo que no es imagen
        raise HTTPException(400, f"no se pudo procesar: {e}")
    return {"items": items, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def medir():
    """Latencia p50/p95 y req/s con TestClient (incluye HTTP en memoria, sin red)."""
    from fastapi.testclient import TestClient
    import torch
    ids = None
    with TestClient(app) as c:
        ids = [x["cliente_id"] for x in c.get("/clientes").json()]
        foto = open(f"data/raw/imagenes/{os.listdir('data/raw/imagenes')[0]}", "rb").read()
        casos = [("/recomendar", v, 200, lambda i, v=v: c.get(f"/recomendar/{ids[i % len(ids)]}?version={v}"))
                 for v in "ABC"]
        casos.append(("/similares (foto)", "-", 20, lambda i: c.post("/similares", files={"foto": ("f.jpg", foto)})))
        filas = []
        for nombre, v, n, llamar in casos:
            llamar(0)  # calentamiento (en /similares carga CLIP)
            ms = []
            for i in range(n):
                t0 = time.perf_counter()
                assert llamar(i).status_code == 200
                ms.append((time.perf_counter() - t0) * 1000)
            ms = np.array(ms)
            filas.append(dict(endpoint=nombre, version=v, p50_ms=round(np.percentile(ms, 50), 1),
                              p95_ms=round(np.percentile(ms, 95), 1), req_s=round(1000 / ms.mean(), 1),
                              device="cpu"))
    df = pd.DataFrame(filas)
    df["usd_por_1000"] = (USD_HORA / (df["req_s"] * 3600) * 1000).round(6)  # supuesto 0.03 USD/h
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    if "--medir" in sys.argv:
        medir()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
