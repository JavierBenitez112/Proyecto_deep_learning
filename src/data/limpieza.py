"""
Limpieza del catálogo de Siman GT.

Entrada:  data/raw/catalogo_siman.csv
Salida:   data/processed/catalogo.csv

Agrega columnas pensadas para el mapeo posterior con H&M:
  seccion        -> index_group_name (mujer, hombre, nina, nino, bebe, unisex)
  tipo_prenda    -> product_type_name
  color_base     -> perceived_colour_master_name
  descuento_pct  -> 1 - precio_q / precio_lista_q

Uso:
  python src/data/limpieza.py   (desde la raíz del repo)
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd

ENTRADA = "data/raw/catalogo_siman.csv"
SALIDA = "data/processed/catalogo.csv"
EMBEDDINGS = "data/embeddings/texto.npz"  # depende del orden de las filas del catálogo

# VTEX pone 10,000,000 de precio a los productos sin precio real (el máximo
# real del catálogo es ~3,500), así que se trata como "sin precio".
PRECIO_CENTINELA = 10_000_000

# DECISIÓN ÓPTICA: todo lo que cuelga de "Óptica" es aros oftálmicos (graduados),
# accesorios de lentes (estuches, paños), soluciones y lentes de contacto: no es
# moda y se quita. Los lentes de sol NO están ahí sino en Accesorios/Lentes de sol,
# que sí se conservan. Por eso la regla es: ruta con "optica" y sin "lentes de sol".

# Palabras clave para tipo_prenda. Se prueban sobre cada tramo de la ruta de
# categoría, de la hoja hacia arriba; gana el primer tramo que coincida y dentro
# del tramo la primera palabra de la lista (por eso lo específico va antes).
# Tramos como "Mujer", "Juvenil", "Curvy" o "Deportivo" no coinciden con nada,
# así que se sube al tramo de arriba (ej. Blusas/Juvenil -> blusa). Si ningún
# tramo coincide se prueba con el nombre del producto.
# Ojo con el orden: "jumpsuit" va antes de "suit" y "trajes de bano" (plural) se
# lista aparte de "traje de bano", si no caen en traje.
# Tallas de criterio: tenis incluye zapato deportivo/lifestyle; bolso incluye
# cartera/mochila/neceser; los combos "Suéter y Chumpas/Chaquetas" de niños son
# casi todos suéteres o sudaderas, así que van como sueter.
TIPOS = [
    ("tenis", "tenis"), ("sneaker", "tenis"), ("zapato deportivo", "tenis"),
    ("zapato lifestyle", "tenis"), ("accesorios para zapatos", "accesorio zapato"),
    ("tacon", "tacon"), ("sandalia", "sandalia"), ("bota", "bota"),
    ("pantufla", "pantufla"), ("zapato", "zapato"), ("zapatilla", "zapato"),
    ("flats", "zapato"), ("loafer", "zapato"), ("mocasin", "zapato"),
    ("zueco", "zapato"), ("mules", "zapato"),
    ("camison", "camison"), ("babydoll", "babydoll"), ("pijama", "pijama"),
    ("blusa", "blusa"), ("camiseta", "camiseta"), ("camisa", "camisa"), ("polo", "polo"),
    ("jean", "jeans"), ("calzoneta", "traje de bano"), ("traje de bano", "traje de bano"),
    ("trajes de bano", "traje de bano"),
    ("leggin", "legging"), ("pantalon", "pantalon"), ("pants", "pantalon"),
    ("jogger", "pantalon"), ("short", "short"), ("bermuda", "short"),
    ("abrigo", "abrigo"), ("sueter", "sueter"), ("sudadera", "sudadera"),
    ("hoodie", "sudadera"), ("chumpa", "chaqueta"), ("chaqueta", "chaqueta"),
    ("chaleco", "chaleco"), ("vestido", "vestido"), ("falda", "falda"),
    ("blazer", "blazer"), ("jumpsuit", "jumpsuit"), ("traje", "traje"), ("suit", "traje"),
    ("top deportivo", "top"), ("conjunto", "conjunto"),
    ("mameluco", "mameluco"), ("overall", "overol"), ("estuche", "estuche"),
    ("panti", "panti"), ("bloomer", "panti"), ("brasier", "brasier"),
    ("brassier", "brasier"), ("faja", "faja"), ("fustan", "faja"),
    ("boxer", "boxer"), ("calzoncillo", "boxer"), ("calcetin", "calcetin"),
    ("media", "calcetin"), ("ropa interior", "ropa interior"),
    ("equipos deportivos", "equipo deportivo"),
    ("bolso", "bolso"), ("cartera", "bolso"), ("neceser", "bolso"), ("mochila", "bolso"),
    ("billetera", "billetera"), ("lentes de sol", "lentes de sol"), ("reloj", "reloj"),
    ("anillo", "anillo"), ("arete", "arete"), ("collar", "collar"), ("cadena", "collar"),
    ("pulsera", "pulsera"), ("joyeria", "joyeria"), ("bisuteria", "joyeria"),
    ("cincho", "cinturon"), ("cinturon", "cinturon"), ("corbata", "corbata"),
    ("tirante", "complemento formal"), ("mancuerna", "complemento formal"),
    ("complementos formales", "complemento formal"),
    ("bufanda", "bufanda"), ("panuelo", "panuelo"), ("sombrilla", "sombrilla"),
    ("gorra", "gorra"), ("sombrero", "gorra"), ("cabello", "accesorio cabello"),
]

# "Estuches para bebé" son sets: el tipo real sale del nombre del producto y gana
# la primera palabra de la lista (un set de baberos y calcetines es un babero).
# "gorrit" cubre gorritos; gorro, gorritos y gorras son todos tipo gorra.
ESTUCHES = [("babero", "babero"), ("salivero", "babero"), ("saliber", "babero"),
            ("gorro", "gorra"), ("gorrit", "gorra"), ("calcetin", "calcetin"),
            ("guante", "guante"), ("conjunto", "conjunto"), ("regalo", "conjunto")]

# Palabras clave (prefijos, sin tildes) para color_base. Gana la que aparece
# primero en el texto ("Azul marino/Blanco" -> azul); estampados y rayas cuentan
# como multicolor. Ejemplos de lo que aparece en el CSV: Navy, Burgundy, Ivory,
# Taupe, Nude, Khaki, Charcoal, Rose Gold, Bicolor_1, Plateado, Dorado, Olivo.
COLORES = {
    "negro": ["negr", "black", "onix", "carbon", "charcoal"],
    "blanco": ["blanc", "white", "off white", "ivory", "marfil"],
    "gris": ["gris", "grey", "gray", "plomo", "jaspe", "marengo"],
    "azul": ["azul", "navy", "marino", "celeste", "turquesa", "aqua", "denim",
             "indigo", "cielo"],
    "rojo": ["roj", "red", "vino\\b", "borgona", "burgundy", "guinda", "tinto"],
    "rosado": ["rosad", "rosa\\b", "pink", "fucsia", "fuchsia", "coral\\b",
               "salmon", "magenta"],
    "verde": ["verde", "green", "oliv", "menta", "militar"],
    "amarillo": ["amarill", "yellow", "mostaza", "limon"],
    "naranja": ["naranja", "anaranjad", "orange", "durazno", "melon\\b",
                "mandarina", "terracota"],
    "morado": ["morad", "purple", "violeta", "lila\\b", "lavanda", "lavender",
               "uva\\b", "purpura"],
    "cafe": ["cafe", "brown", "marron", "chocolate", "tabaco", "canela", "tostado"],
    "beige": ["beige", "crema", "arena\\b", "nude\\b", "hueso", "natural\\b",
              "crudo", "ecru", "caqui", "khaki", "taupe", "camel"],
    "multicolor": ["multicolor", "bicolor", "tricolor", "estampad", "print",
                   "rayad", "rayas", "cuadros", "floral", "camuflaj", "tie dye",
                   "leopardo", "surtido"],
    "metalico": ["dorad", "oro\\b", "gold", "platead", "plata\\b", "silver",
                 "metalico", "bronce", "cobre"],
}
PATRONES_COLOR = {c: re.compile(r"\b(?:" + "|".join(p) + ")") for c, p in COLORES.items()}


def sin_tildes(t):
    """Minúsculas y sin tildes (la ñ pasa a n): 'Niñas' -> 'ninas'."""
    t = unicodedata.normalize("NFD", str(t))
    return "".join(c for c in t if unicodedata.category(c) != "Mn").lower().strip()


def seccion_de(texto):
    """Sección desde una ruta de categoría (o un nombre): bebe gana sobre nina/nino."""
    t = sin_tildes(texto)
    if re.search(r"\bbebe|toddler", t):
        return "bebe"
    if re.search(r"\bnina", t):
        return "nina"
    if re.search(r"\bnino", t):
        return "nino"
    if re.search(r"\b(mujer|mujeres|damas?)\b", t):
        return "mujer"
    if re.search(r"\b(hombres?|caballeros?)\b", t):
        return "hombre"
    return "unisex"


def seccion_producto(categoria, nombre):
    """Por la ruta; si no alcanza (ej. Moda/Accesorios), por el nombre del producto."""
    s = seccion_de(categoria)
    return s if s != "unisex" else seccion_de(nombre)


def tipo_estuche(nombre):
    """Tipo real de un set de bebé por su nombre; 'estuche' si nada coincide."""
    n = sin_tildes(nombre)
    return next((tipo for palabra, tipo in ESTUCHES if palabra in n), "estuche")


def tipo_de(categoria, nombre):
    tramos = [sin_tildes(t) for t in str(categoria).split("/") if t.strip()]
    # En "Ropa de dormir" la hoja "Conjunto" (y similares) es un pijama
    if "dormir" in sin_tildes(categoria) and not re.search(r"camison|babydoll", tramos[-1]):
        return "pijama"
    for tramo in tramos[::-1] + [sin_tildes(nombre)]:
        for palabra, tipo in TIPOS:
            if palabra in tramo:
                return tipo_estuche(nombre) if tipo == "estuche" else tipo
    return tramos[-1] if tramos else "otro"


def color_de(texto):
    t = sin_tildes(texto)
    mejor, pos = "otro", len(t) + 1
    for color, patron in PATRONES_COLOR.items():
        m = patron.search(t)
        if m and m.start() < pos:
            mejor, pos = color, m.start()
    return mejor


def color_producto(color, nombre):
    """Por la columna color; si es 'Único' u otro texto sin color, por el nombre."""
    c = color_de(color)
    return c if c != "otro" else color_de(nombre)


def filtrar(df, motivo, mantener, quitadas):
    """Se queda con las filas donde mantener es True y anota cuántas quitó."""
    quitadas[motivo] = int((~mantener).sum())
    return df[mantener]


def main():
    df = pd.read_csv(ENTRADA, dtype={"product_id": str, "item_id": str})
    antes = len(df)
    quitadas = {}

    # 1. Filas inválidas, duplicados y óptica
    df["precio_q"] = pd.to_numeric(df["precio_q"], errors="coerce")
    df["precio_lista_q"] = pd.to_numeric(df["precio_lista_q"], errors="coerce")
    sin_imagen = df["imagen_url"].isna() | (df["imagen_url"].astype(str).str.strip() == "")
    df = filtrar(df, "sin imagen", ~sin_imagen, quitadas)
    df = filtrar(df, "sin precio o precio <= 0", df["precio_q"] > 0, quitadas)
    df = filtrar(df, f"precio centinela ({PRECIO_CENTINELA:,})",
                 df["precio_q"] < PRECIO_CENTINELA, quitadas)
    df = filtrar(df, "item_id duplicado", ~df.duplicated("item_id"), quitadas)
    df = filtrar(df, "product_id+color duplicado",
                 ~df.duplicated(["product_id", "color"]), quitadas)
    ruta = df["categoria"].map(sin_tildes)
    df = filtrar(df, "óptica (no es moda, ver DECISIÓN ÓPTICA)",
                 ~(ruta.str.contains("optica") & ~ruta.str.contains("lentes de sol")), quitadas)

    # 2. Columnas nuevas
    df = df.copy()
    df["seccion"] = [seccion_producto(c, n) for c, n in zip(df["categoria"], df["nombre"])]
    df["tipo_prenda"] = [tipo_de(c, n) for c, n in zip(df["categoria"], df["nombre"])]
    df["color_base"] = [color_producto(c, n) for c, n in zip(df["color"], df["nombre"])]
    lista = df["precio_lista_q"].where(df["precio_lista_q"] < PRECIO_CENTINELA)
    df["descuento_pct"] = (1 - df["precio_q"] / lista).where(lista > 0, 0).clip(lower=0).round(4)

    # El orden de las filas no puede cambiar: texto.npz está alineado con él
    if os.path.exists(EMBEDDINGS):
        ids = np.load(EMBEDDINGS)["item_id"].astype(str).tolist()
        assert ids == df["item_id"].tolist(), "item_id distinto (o en otro orden) que texto.npz"

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    df.to_csv(SALIDA, index=False, encoding="utf-8")

    print(f"Filas antes: {antes}   después: {len(df)}")
    for motivo, n in quitadas.items():
        print(f"  quitadas por {motivo}: {n}")
    print("\nPor sección:\n" + df["seccion"].value_counts().to_string())
    print("\nTop 15 tipo_prenda:\n" + df["tipo_prenda"].value_counts().head(15).to_string())
    print("\nPor color_base:\n" + df["color_base"].value_counts().to_string())
    print(f"\n% color_base 'otro': {(df['color_base'] == 'otro').mean():.1%}")

    assert df["imagen_url"].notna().all() and df["precio_q"].notna().all()
    assert df["item_id"].is_unique
    assert df["seccion"].notna().all() and df["color_base"].notna().all()
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
