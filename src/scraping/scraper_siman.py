"""
Scraper del catálogo de moda de Siman Guatemala (gt.siman.com).
Usa la API pública de VTEX, así que no hay que leer HTML.

Salida:
  data/raw/catalogo_siman.csv   -> una fila por producto/color
  data/raw/imagenes/        -> imagen principal de cada producto (opcional)

Uso:
  pip install requests pandas
  python src/scraping/scraper_siman.py   (desde la raíz del repo)
"""

import os
import re
import time
import requests
import pandas as pd

BASE = "https://gt.siman.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico UVG - CC3092)"}
PAUSA = 1.0              # segundos entre peticiones (no saturar el servidor)
PAGINA = 50              # VTEX permite máximo 50 por página
MAX_POR_CATEGORIA = 2500 # VTEX no devuelve más de 2500 resultados por búsqueda
DESCARGAR_IMAGENES = True

# Palabras para quedarnos solo con categorías de moda
PALABRAS_MODA = ["ropa", "moda", "calzado", "zapato", "damas", "caballeros",
                 "mujer", "hombre", "jeans", "vestido", "blusa", "camisa",
                 "pantal", "accesorios de moda", "bolso", "cartera"]
EXCLUIR = ["juguete", "belleza", "hogar", "electr", "tecnolog", "cosm",
           "mascota", "farmacia"]


def get(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(PAUSA)
    return r.json()


def categorias_moda():
    """Recorre el árbol de categorías y devuelve las hojas que son de moda."""
    arbol = get(f"{BASE}/api/catalog_system/pub/category/tree/3")
    hojas = []

    def recorrer(nodo, ruta, ids):
        ruta = ruta + [nodo["name"]]
        ids = ids + [str(nodo["id"])]
        hijos = nodo.get("children", [])
        if not hijos:
            texto = " / ".join(ruta).lower()
            if any(p in texto for p in PALABRAS_MODA) and not any(e in texto for e in EXCLUIR):
                # Siman solo filtra con la ruta completa de IDs: C:/41/141/240/
                hojas.append({"fq": f"C:/{'/'.join(ids)}/", "ruta": " / ".join(ruta)})
        for h in hijos:
            recorrer(h, ruta, ids)

    for raiz in arbol:
        recorrer(raiz, [], [])
    return hojas


def productos_de_categoria(fq):
    """Descarga todos los productos de una categoría, paginando."""
    productos, desde = [], 0
    while desde < MAX_POR_CATEGORIA:
        params = {"fq": fq, "_from": desde, "_to": desde + PAGINA - 1}
        try:
            lote = get(f"{BASE}/api/catalog_system/pub/products/search", params)
        except requests.HTTPError:
            break
        if not lote:
            break
        productos.extend(lote)
        if len(lote) < PAGINA:
            break
        desde += PAGINA
    return productos


def limpiar_texto(t):
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", t).strip()


def aplanar(p, ruta_cat):
    """Convierte un producto VTEX en filas (una por color)."""
    filas, vistos = [], set()
    for item in p.get("items", []):
        color = (item.get("Color") or ["Único"])[0]
        if color in vistos:
            continue
        vistos.add(color)
        oferta = item["sellers"][0]["commertialOffer"] if item.get("sellers") else {}
        imgs = item.get("images", [])
        filas.append({
            "product_id": p["productId"],
            "item_id": item["itemId"],
            "nombre": p.get("productName"),
            "marca": p.get("brand"),
            "categoria": (p.get("categories") or [ruta_cat])[0].strip("/"),
            "color": color,
            "descripcion": limpiar_texto(p.get("description")),
            "precio_q": oferta.get("Price"),
            "precio_lista_q": oferta.get("ListPrice"),
            "disponible": oferta.get("IsAvailable"),
            "imagen_url": imgs[0]["imageUrl"] if imgs else None,
            "link": p.get("link"),
            "fecha_lanzamiento": p.get("releaseDate"),
        })
    return filas


def descargar_imagen(url, destino):
    if os.path.exists(destino):
        return
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(destino, "wb") as f:
            f.write(r.content)
        time.sleep(PAUSA / 2)
    except requests.RequestException:
        pass


def main():
    os.makedirs("data/raw/imagenes", exist_ok=True)

    cats = categorias_moda()
    print(f"Categorías de moda encontradas: {len(cats)}")

    filas = []
    for i, c in enumerate(cats, 1):
        prods = productos_de_categoria(c["fq"])
        for p in prods:
            filas.extend(aplanar(p, c["ruta"]))
        print(f"[{i}/{len(cats)}] {c['ruta']}: {len(prods)} productos")

    df = pd.DataFrame(filas).drop_duplicates(subset=["item_id"])
    df.to_csv("data/raw/catalogo_siman.csv", index=False, encoding="utf-8")
    print(f"\nTotal: {len(df)} filas guardadas en data/raw/catalogo_siman.csv")

    if DESCARGAR_IMAGENES:
        print("Descargando imágenes...")
        for _, row in df.dropna(subset=["imagen_url"]).iterrows():
            descargar_imagen(row["imagen_url"], f"data/raw/imagenes/{row['item_id']}.jpg")
        print("Listo.")


if __name__ == "__main__":
    main()
