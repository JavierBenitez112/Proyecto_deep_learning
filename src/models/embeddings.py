"""
Embeddings de texto del catálogo de Siman GT.

Entrada:  data/processed/catalogo.csv
Salida:   data/embeddings/texto.npz
            item_id -> ids en el mismo orden que el CSV (texto)
            emb     -> float32, N x 768, normalizados (norma 1)

Modelo: intfloat/multilingual-e5-base (los textos están en español). Se descarga
de Hugging Face la primera vez (~1.1 GB).

Con --imagen se calculan en cambio los embeddings de imagen con CLIP:
Entrada:  data/processed/catalogo.csv + data/raw/imagenes/{item_id}.jpg
Salida:   data/embeddings/imagen.npz
            item_id      -> ids en el mismo orden que el CSV (y que texto.npz)
            emb          -> float32, N x 512, norma 1 (ceros si no hay imagen)
            tiene_imagen -> bool, False si la foto falta o no abre
          reports/figuras/similares_imagen.png (prenda consulta + 5 vecinos)
Modelo: openai/clip-vit-base-patch32 (~600 MB la primera vez).

El EDA encontró que el 39 % de las filas comparte descripción, así que el
nombre y la categoría van al inicio del texto y la descripción al final.

Uso:
  python src/models/embeddings.py            # texto
  python src/models/embeddings.py --imagen   # imagen (CLIP)
(desde la raíz del repo)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

ENTRADA = "data/processed/catalogo.csv"
SALIDA = "data/embeddings/texto.npz"
MODELO = "intfloat/multilingual-e5-base"
BATCH = 64
MAX_PALABRAS = 200  # la descripción se recorta a esto
SEMILLA = 42
SALIDA_IMG = "data/embeddings/imagen.npz"
MODELO_IMG = "openai/clip-vit-base-patch32"
CARPETA_IMG = "data/raw/imagenes"
FIGURA = "reports/figuras/similares_imagen.png"


def texto_prenda(fila):
    """Texto de una prenda: 'passage: nombre. categoria. Color: x. descripcion'."""
    desc = " ".join(str(fila["descripcion"]).split()[:MAX_PALABRAS])
    return (f"passage: {fila['nombre']}. {fila['categoria']}. "
            f"Color: {fila['color_base']}. {desc}")


def vecinos(emb, i, k=5):
    """Índices de los k vecinos más cercanos por coseno (embeddings ya normalizados)."""
    sim = emb @ emb[i]
    sim[i] = -np.inf  # excluir la prenda misma
    return np.argsort(-sim)[:k]


def _abrir(ruta):
    """Imagen RGB o None si falta / no abre."""
    try:
        return Image.open(ruta).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def _clip(imagenes, modelo, procesador, device):
    """Vectores CLIP normalizados (float32, n x 512) de una lista de imágenes PIL."""
    x = procesador(images=imagenes, return_tensors="pt")["pixel_values"].to(device)
    v = modelo.get_image_features(pixel_values=x)
    v = getattr(v, "pooler_output", v)  # transformers 5 puede devolver un objeto
    return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float32)


def emb_foto(ruta, modelo, procesador, device):
    """Vector CLIP normalizado (512,) de una foto cualquiera (para cold start en la API)."""
    return _clip([Image.open(ruta).convert("RGB")], modelo, procesador, device)[0]


def main_imagen():
    df = pd.read_csv(ENTRADA, dtype={"item_id": str})
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    modelo = CLIPModel.from_pretrained(MODELO_IMG).to(device).eval()
    procesador = CLIPProcessor.from_pretrained(MODELO_IMG)

    ids = df["item_id"].tolist()
    emb = np.zeros((len(df), 512), dtype=np.float32)
    tiene = np.zeros(len(df), dtype=bool)
    t0 = time.time()
    for ini in range(0, len(df), BATCH):
        fotos = [_abrir(f"{CARPETA_IMG}/{i}.jpg") for i in ids[ini:ini + BATCH]]
        ok = [k for k, f in enumerate(fotos) if f is not None]
        if ok:
            emb[[ini + k for k in ok]] = _clip([fotos[k] for k in ok], modelo, procesador, device)
            tiene[[ini + k for k in ok]] = True
        if (ini // BATCH) % 20 == 0:
            print(f"  {ini}/{len(df)}", flush=True)
    segundos = time.time() - t0

    os.makedirs(os.path.dirname(SALIDA_IMG), exist_ok=True)
    np.savez(SALIDA_IMG, item_id=df["item_id"].to_numpy(dtype=str), emb=emb, tiene_imagen=tiene)
    print(f"\nForma: {emb.shape}   device: {device}   tiempo de cómputo: {segundos:.1f} s")
    print(f"Sin imagen: {(~tiene).sum()} de {len(df)}")

    # Prueba de emb_foto: la foto de un item debe tener a sí mismo como vecino #1
    i = int(np.flatnonzero(tiene)[0])
    v = emb_foto(f"{CARPETA_IMG}/{ids[i]}.jpg", modelo, procesador, device)
    sim = emb @ v
    assert sim[i] > 0.99 and int(np.argmax(sim)) == i, sim[i]
    print(f"emb_foto: similitud consigo misma {sim[i]:.4f}")

    # Figura de cold start: consulta + 5 vecinos, para mujer / hombre / niño-bebé
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEMILLA)
    con_foto = df.index[tiene]
    filas = [int(rng.choice(con_foto[df.loc[con_foto, "seccion"].isin(sec)]))
             for sec in (["mujer"], ["hombre"], ["nino", "bebe"])]
    fig, ejes = plt.subplots(3, 6, figsize=(15, 8))
    for r, q in enumerate(filas):
        # vecinos() de abajo no filtra prendas sin foto (vector cero): sim 0, quedan al final
        for c, j in enumerate([q] + list(vecinos(emb, q))):
            ax = ejes[r, c]
            ax.imshow(_abrir(f"{CARPETA_IMG}/{ids[j]}.jpg"))
            ax.axis("off")
            t = df.at[j, "nombre"][:25]
            ax.set_title(f"CONSULTA\n{t}" if c == 0 else f"{t}\n{float(emb[q] @ emb[j]):.3f}", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIGURA), exist_ok=True)
    fig.savefig(FIGURA, dpi=110)
    print(f"Figura en {FIGURA}")

    # Comprobaciones
    assert emb.shape == (len(df), 512) and emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert np.allclose(np.linalg.norm(emb[tiene], axis=1), 1.0, atol=1e-3)
    assert (emb[~tiene] == 0).all()
    assert (np.load(SALIDA)["item_id"] == np.array(ids)).all()  # igual que texto.npz
    muestra = rng.choice(con_foto, 300, replace=False)
    tipos = df["tipo_prenda"].to_numpy()
    pct = np.mean([np.mean(tipos[vecinos(emb, i)] == tipos[i]) for i in muestra])
    print(f"% de vecinos (top 5) con el mismo tipo_prenda, en 300 prendas: {pct:.1%}")
    assert pct > 0.4
    print(f"Guardado en {SALIDA_IMG}")


def main():
    df = pd.read_csv(ENTRADA, dtype={"item_id": str})
    textos = [texto_prenda(f) for _, f in df.iterrows()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    modelo = SentenceTransformer(MODELO, device=device)
    t0 = time.time()
    emb = modelo.encode(textos, batch_size=BATCH, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    segundos = time.time() - t0

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    np.savez(SALIDA, item_id=df["item_id"].to_numpy(dtype=str), emb=emb)

    print(f"\nForma: {emb.shape}   device: {device}   tiempo de cómputo: {segundos:.1f} s")

    # Vecinos de ejemplo: una prenda de mujer, una de hombre y una de niño/bebé
    rng = np.random.default_rng(SEMILLA)
    for secciones in (["mujer"], ["hombre"], ["nino", "bebe"]):
        i = int(rng.choice(df.index[df["seccion"].isin(secciones)]))
        print(f"\n[{df.at[i, 'seccion']}] {df.at[i, 'nombre']}  ({df.at[i, 'tipo_prenda']})")
        for j in vecinos(emb, i):
            print(f"   {float(emb[i] @ emb[j]):.3f}  {df.at[j, 'nombre']}  ({df.at[j, 'tipo_prenda']})")

    # Comprobaciones
    assert len(emb) == len(df) and emb.shape[1] == 768 and emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    muestra = rng.choice(len(df), 300, replace=False)
    tipos = df["tipo_prenda"].to_numpy()
    pct = np.mean([np.mean(tipos[vecinos(emb, i)] == tipos[i]) for i in muestra])
    print(f"\n% de vecinos (top 5) con el mismo tipo_prenda, en 300 prendas: {pct:.1%}")
    assert pct > 0.5
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    main_imagen() if "--imagen" in sys.argv else main()
