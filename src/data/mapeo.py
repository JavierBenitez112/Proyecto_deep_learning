"""
Mapeo de los artículos de H&M a prendas del catálogo de Siman GT.

Entrada:  data/processed/transacciones_hm.csv   (de muestreo.py)
          data/raw/hm/articles.csv
          data/processed/catalogo.csv
Salida:   data/processed/mapeo_hm_siman.csv   article_id, item_id, nivel, seccion, tipo_prenda, color_base
          data/processed/transacciones.csv    t_dat, customer_id, item_id, precio_q, split
          docs/reglas_mapeo.md                reglas y estadísticas para el informe

Cada article_id de H&M recibe UN item_id de Siman:
  1. Se traduce el artículo a (seccion, tipo_prenda, color_base) con los diccionarios de abajo.
  2. Se buscan candidatos por niveles: (1) seccion+tipo+color, (2) seccion+tipo,
     (3) grupo_edad+tipo (adultos: mujer/hombre/unisex; niños: nina/nino/bebe; nunca se cruzan).
     Las faldas de mujer se mapean a vestidos de mujer (Siman no tiene faldas de mujer).
  3. Dentro de los candidatos (ordenados por precio_q) se elige por cuantil de precio:
     el artículo en el percentil p de su grupo H&M va al item en el índice floor(p·n).
Los artículos sin equivalente en Siman (cosméticos, muebles, etc.) se descartan.

Uso:
  python src/data/mapeo.py   (desde la raíz del repo)
"""

import os
import re
import numpy as np
import pandas as pd

TRANS_HM = "data/processed/transacciones_hm.csv"
ARTICULOS = "data/raw/hm/articles.csv"
CATALOGO = "data/processed/catalogo.csv"
SAL_MAPEO = "data/processed/mapeo_hm_siman.csv"
SAL_TRANS = "data/processed/transacciones.csv"
SAL_DOC = "docs/reglas_mapeo.md"

INICIO_VAL = "2020-09-09"
MIN_CONSERVADAS = 0.80  # fracción mínima de transacciones que debe quedar mapeada
NINOS_POR_DEFECTO = "nina"  # niños sin pista de género (ver pista_ninos)

# Grupo de edad de cada sección: el nivel 3 puede cambiar de sección pero no de grupo
GRUPO_EDAD = {"mujer": "adulto", "hombre": "adulto", "unisex": "adulto",
              "nina": "ninos", "nino": "ninos", "bebe": "ninos"}

# Niveles de candidatos: columnas que deben coincidir entre el artículo y el catálogo
NIVELES = {1: ["seccion", "tipo_prenda", "color_base"],
           2: ["seccion", "tipo_prenda"],
           3: ["grupo_edad", "tipo_prenda"]}

# --- Diccionario 2: product_type_name (H&M) -> tipo_prenda (Siman) ---------------------
# Lo que no aparece aquí no tiene equivalente en Siman y se descarta (cosméticos,
# muebles, papelería, juguetes, textiles de hogar, cuidado de ropa, Unknown, etc.).
# Criterios discutibles: Top -> blusa y Vest top -> camiseta (se reparte la demanda);
# Coat -> chaqueta (Siman casi no tiene abrigos); Cardigan -> sueter; Bodysuit ->
# camiseta; Underwear body/set -> brasier; Underdress -> faja (el fustán de Siman);
# Robe -> camison; Wedge -> sandalia; Underwear Tights -> panti (Siman agrupa las pantimedias).
EQUIVALENCIAS = {
    "accesorio cabello": ["Alice band", "Hair clip", "Hair string", "Hair ties",
                          "Hair/alice band", "Hairband", "Headband"],
    "anillo": ["Ring"],
    "arete": ["Earring", "Earrings"],
    "babero": ["Baby Bib"],
    "billetera": ["Wallet"],
    "blazer": ["Blazer"],
    "blusa": ["Blouse", "Top"],
    "bolso": ["Bag", "Backpack", "Bumbag", "Cross-body bag", "Shoulder bag", "Tote bag",
              "Weekend/Gym bag"],
    "bota": ["Boots", "Bootie"],
    "brasier": ["Bra", "Underwear body", "Underwear set"],
    "bufanda": ["Scarf"],
    "calcetin": ["Socks", "Leg warmers"],
    "camisa": ["Shirt"],
    "camiseta": ["T-shirt", "Vest top", "Bodysuit"],
    "camison": ["Night gown", "Robe"],
    "chaleco": ["Outdoor Waistcoat", "Tailored Waistcoat"],
    "chaqueta": ["Jacket", "Coat"],
    "cinturon": ["Belt"],
    "collar": ["Necklace"],
    "complemento formal": ["Braces"],
    "conjunto": ["Garment Set"],
    "corbata": ["Tie"],
    "faja": ["Underdress"],
    "falda": ["Skirt"],
    "gorra": ["Beanie", "Bucket hat", "Cap", "Cap/peaked", "Felt hat", "Hat/beanie",
              "Hat/brim", "Straw hat"],
    "guante": ["Gloves"],
    "jumpsuit": ["Jumpsuit/Playsuit"],
    "legging": ["Leggings/Tights"],
    "lentes de sol": ["Sunglasses"],
    "overol": ["Dungarees", "Outdoor overall"],
    "pantalon": ["Trousers", "Outdoor trousers"],
    "pantufla": ["Slippers"],
    "panti": ["Underwear bottom", "Underwear Tights"],
    "pijama": ["Pyjama bottom", "Pyjama jumpsuit/playsuit", "Pyjama set", "Sleeping sack",
               "Sleep Bag"],
    "polo": ["Polo shirt"],
    "pulsera": ["Bracelet"],
    "reloj": ["Watch"],
    "ropa interior": ["Kids Underwear top", "Long John"],
    "sandalia": ["Sandals", "Heeled sandals", "Flip flop", "Wedge"],
    "short": ["Shorts"],
    "sombrilla": ["Umbrella"],
    "sudadera": ["Hoodie"],
    "sueter": ["Sweater", "Cardigan"],
    "tacon": ["Heels", "Pumps"],
    "tenis": ["Sneakers"],
    "traje de bano": ["Bikini top", "Swimsuit", "Swimwear bottom", "Swimwear set",
                      "Swimwear top"],
    "vestido": ["Dress"],
    "zapato": ["Ballerinas", "Flat shoe", "Flat shoes", "Moccasins", "Other shoe",
               "Pre-walkers"],
}
TIPO = {h: s for s, hs in EQUIVALENCIAS.items() for h in hs}

# Ajustes por sección: Siman clasifica distinto según la sección (ej. las camisetas de
# niña están como "blusa", los zapatos de niños como "zapato"). Solo se usan donde el
# tipo directo casi no existe en esa sección del catálogo.
# (tipo traducido, sección) -> tipo de Siman
AJUSTES = {}
for _tipo, _secciones, _nuevo in [
    ("camiseta", ["nina"], "blusa"),
    ("camisa", ["mujer", "nina"], "blusa"),
    ("blusa", ["hombre"], "camiseta"),
    ("polo", ["mujer", "nino", "bebe"], "camiseta"),
    ("polo", ["nina"], "blusa"),
    ("tenis", ["nina", "nino", "bebe"], "zapato"),
    ("bota", ["nina", "nino", "bebe"], "zapato"),
    ("sandalia", ["nina", "nino", "bebe"], "zapato"),
    ("chaqueta", ["nina", "nino", "bebe"], "sueter"),
    ("chaqueta", ["hombre"], "sudadera"),
    ("sudadera", ["mujer", "nina", "nino", "bebe"], "sueter"),
    ("panti", ["hombre", "nino"], "boxer"),
    ("legging", ["nina", "bebe"], "pantalon"),
    ("jumpsuit", ["bebe"], "mameluco"),
    ("jumpsuit", ["nina"], "vestido"),
    ("falda", ["mujer"], "vestido"),
]:
    for _s in _secciones:
        AJUSTES[(_tipo, _s)] = _nuevo

# --- Diccionario 3: perceived_colour_master_name (H&M) -> color_base (Siman) -----------
COLOR = {
    "Black": "negro", "White": "blanco", "Grey": "gris", "Blue": "azul", "Turquoise": "azul",
    "Red": "rojo", "Pink": "rosado", "Green": "verde", "Khaki green": "verde",
    "Yellowish Green": "verde", "Bluish Green": "verde", "Yellow": "amarillo",
    "Orange": "naranja", "Lilac Purple": "morado", "Brown": "cafe", "Beige": "beige",
    "Mole": "beige", "Metal": "metalico", "Unknown": "otro", "undefined": "otro",
}
# graphical_appearance_name que cuenta como estampado/rayas -> multicolor. Se dejan fuera
# Front print y Placement print: son un dibujo sobre una prenda de color liso.
MULTICOLOR = {"All over pattern", "Stripe", "Check", "Dot", "Argyle", "Other pattern",
              "Mixed solid/pattern", "Colour blocking"}


def pista_ninos(seccion, depto, nombre):
    """'nina' o 'nino' según la primera pista con Girl/Boy (sección, departamento, nombre)."""
    for texto in (seccion, depto, nombre):
        t = str(texto).lower()
        if ("girl" in t) != ("boy" in t):
            return "nina" if "girl" in t else "nino"
    return None


def seccion_hm(grupo, indice, seccion, depto, nombre):
    """Diccionario 1: sección de Siman desde index_group_name, index_name y section_name."""
    if indice.startswith("Baby Sizes"):
        return "bebe"
    if grupo == "Baby/Children" or seccion == "Kids Sports":
        return pista_ninos(seccion, depto, nombre) or NINOS_POR_DEFECTO
    if grupo == "Menswear":
        return "hombre"
    if grupo in ("Divided", "Sport") and re.search(r"\bmens?\b", seccion.lower()):
        return "hombre"
    return "mujer"  # Ladieswear, Lingeries, Ladies Accessories, Divided y Sport sin "Men"


def tipo_hm(producto, seccion_hm_name, apariencia, seccion):
    """Tipo de Siman (o None si no hay equivalente), con jeans por Denim y ajustes por sección."""
    tipo = TIPO.get(producto)
    if producto == "Trousers" and (apariencia == "Denim" or "Denim" in seccion_hm_name):
        tipo = "jeans"
    return AJUSTES.get((tipo, seccion), tipo)


def color_hm(color, apariencia):
    return "multicolor" if apariencia in MULTICOLOR else COLOR.get(color, "otro")


def preparar_articulos(ids, precios):
    """Una fila por artículo de la muestra con seccion, tipo_prenda, color_base y precio medio."""
    cols = ["article_id", "product_type_name", "graphical_appearance_name",
            "perceived_colour_master_name", "index_group_name", "index_name",
            "section_name", "department_name", "prod_name"]
    art = pd.read_csv(ARTICULOS, usecols=cols, dtype=str)
    art = art[art["article_id"].isin(ids)].sort_values("article_id", ignore_index=True)
    art["seccion"] = [seccion_hm(*f) for f in zip(
        art["index_group_name"], art["index_name"], art["section_name"],
        art["department_name"], art["prod_name"])]
    art["tipo_prenda"] = [tipo_hm(*f) for f in zip(
        art["product_type_name"], art["section_name"], art["graphical_appearance_name"],
        art["seccion"])]
    art["color_base"] = [color_hm(*f) for f in zip(
        art["perceived_colour_master_name"], art["graphical_appearance_name"])]
    art["precio"] = art["article_id"].map(precios)
    return art


def llave(df, claves):
    return df[claves].astype(str).agg("|".join, axis=1)


def asignar(art, cat):
    """Agrega item_id ('' si no hay) y nivel (0 si no hay) a cada artículo, por niveles."""
    # la lista de candidatos queda por precio
    cat = cat.assign(grupo_edad=cat["seccion"].map(GRUPO_EDAD)).sort_values(["precio_q", "item_id"])
    art = art.assign(grupo_edad=art["seccion"].map(GRUPO_EDAD), item_id="", nivel=0)
    for nivel, claves in NIVELES.items():
        cand = cat.assign(k=llave(cat, claves)).groupby("k")["item_id"].agg(list)
        pend = art[(art["nivel"] == 0) & art["tipo_prenda"].notna()]
        pend = pend.assign(k=llave(pend, claves))
        pend = pend[pend["k"].isin(cand.index)]
        # percentil de precio dentro del grupo H&M que comparte la misma lista de candidatos
        g = pend.groupby("k")["precio"]
        p = (g.rank(method="first") - 0.5) / g.transform("size")
        n = pend["k"].map(cand.str.len())
        pos = np.minimum(np.floor(p * n).astype(int), n - 1)
        art.loc[pend.index, "item_id"] = [cand[k][i] for k, i in zip(pend["k"], pos)]
        art.loc[pend.index, "nivel"] = nivel
    return art


def md_tabla(df):
    """DataFrame -> tabla markdown (el índice es la primera columna)."""
    def celda(v):
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    filas = [[df.index.name or ""] + list(df.columns)]
    filas += [[i] + list(r) for i, r in df.astype(object).iterrows()]
    cuerpo = [filas[0], ["---"] * len(filas[0])] + [[celda(v) for v in f] for f in filas[1:]]
    return "\n".join("| " + " | ".join(map(str, f)) + " |" for f in cuerpo)


def pct(x, total):
    return f"{x / total:.1%}"


def escribir_doc(stats):
    """Arma docs/reglas_mapeo.md con las reglas y las estadísticas de esta corrida."""
    dic_tipos = "\n".join(f"| {s} | {', '.join(hs)} |" for s, hs in sorted(EQUIVALENCIAS.items()))
    ajustes = "\n".join(f"| {t} | {s} | {n} |" for (t, s), n in sorted(AJUSTES.items()))
    colores = "\n".join(f"| {c} | {', '.join(h for h, v in COLOR.items() if v == c)} |"
                        for c in sorted(set(COLOR.values())))
    texto = f"""# Reglas del mapeo H&M → Siman

Las compras del proyecto son las de H&M (últimos 6 meses, {stats['clientes']:,} clientes), pero el
sistema recomienda prendas de Siman GT. **Supuesto central:** quien compra un artículo de H&M
compraría "la misma prenda" (misma sección, tipo y color, y precio en la misma posición relativa) en
Siman. Cada artículo de H&M se asigna a **un solo** item de Siman. Código: `src/data/mapeo.py`.

## Reglas

1. **Traducción del artículo** a tres atributos, con los diccionarios de abajo: `seccion`, `tipo_prenda`
   y `color_base` (los mismos valores del catálogo de Siman).
2. **Candidatos por niveles** (se usa el primer nivel con candidatos):

   | Nivel | Candidatos del catálogo |
   |---|---|
   | 1 | misma `seccion` + `tipo_prenda` + `color_base` |
   | 2 | misma `seccion` + `tipo_prenda` |
   | 3 | mismo `tipo_prenda` y mismo grupo de edad (adulto: mujer, hombre, unisex; niños: nina, nino, bebe), cualquier sección |
   | – | sin candidatos: el artículo queda sin mapear |

3. **Elección dentro de los candidatos (cuantiles de precio).** Se calcula el precio medio de cada
   artículo H&M en la muestra y se ordenan los artículos del grupo (los que comparten nivel y lista de
   candidatos). Con el percentil p = (posición − 0.5) / tamaño del grupo, el artículo va al item en el
   índice `floor(p·n)` de la lista de n candidatos ordenada por `precio_q` (empates por `item_id`).
   Conserva el orden de precios, reparte las compras entre todos los candidatos y es determinista.
4. **Se descartan** las transacciones de artículos sin equivalente en Siman o sin candidatos.
5. El precio de la transacción es el `precio_q` del item de Siman, no el de H&M.

## Diccionarios (resumen)

**Sección** (`index_group_name`, `index_name`, `section_name`):

| H&M | seccion |
|---|---|
| Baby Sizes 50-98 | bebe |
| Ladieswear, Lingeries/Tights, Ladies Accessories | mujer |
| Menswear | hombre |
| Baby/Children (resto) y Kids Sports | nina / nino: palabra Girl/Boy en section_name, luego en department_name, luego en prod_name. Sin pista ({stats['ninos_defecto']:,} artículos) → **{NINOS_POR_DEFECTO}** |
| Divided, Sport | hombre si section_name dice Men; si no, mujer |

**Tipo** (`product_type_name`, unos 130 valores, → tipo de Siman):

| tipo_prenda (Siman) | product_type_name (H&M) |
|---|---|
{dic_tipos}

Sin equivalente (se descartan): {', '.join(stats['sin_equivalente'])}.

Además, `Trousers` con apariencia Denim (o sección Denim) → **jeans**. Y estos ajustes por sección,
porque Siman clasifica distinto según la sección:

| tipo traducido | seccion | tipo final |
|---|---|---|
{ajustes}

**Color** (`perceived_colour_master_name` → `color_base`). Si `graphical_appearance_name` es
{', '.join(sorted(MULTICOLOR))}, el color es **multicolor**.

| color_base | perceived_colour_master_name |
|---|---|
{colores}

## Estadísticas

Muestra: {stats['filas']:,} transacciones y {stats['articulos']:,} artículos de H&M.

**Por nivel de mapeo**

{md_tabla(stats['por_nivel'])}

**Transacciones descartadas:** {stats['desc_n']:,} ({stats['desc_pct']}). Conservadas: {stats['cons_pct']}.

{md_tabla(stats['descartes'])}

**Items de Siman que reciben al menos 1 compra:** {stats['items_con_compra']:,} de {stats['items_total']:,}
({stats['items_pct']}); de los {stats['items_disp']:,} disponibles, {stats['items_disp_con']:,}.

**Concentración:** los 10 items más comprados se llevan el {stats['top10']} de las compras y los
100 más comprados el {stats['top100']}.

**Compras por sección y tipo con más transacciones en nivel 3** (la sección de H&M no tiene ese tipo en
Siman, así que la prenda cae en otra sección del mismo grupo de edad):

{md_tabla(stats['nivel3'])}

## Limitaciones

- **Es un supuesto, no una observación.** Nadie compró esos items de Siman; el mapeo traslada el patrón
  de compra de H&M (Europa, 2020) a otro catálogo, otro país y otra época. Las métricas miden qué tan bien
  se recupera ese patrón simulado, no ventas reales en Guatemala.
- **Un artículo → un item.** Se pierde la variedad: todos los clientes que compraron el mismo artículo
  de H&M "compraron" el mismo item de Siman.
- **Concentración por falta de oferta.** Donde Siman tiene pocos items de un tipo y sección (trajes de
  baño de mujer, sudaderas, leggings), muchas compras caen en pocos items. Esto favorece a la línea base
  de más vendidos.
- **Nivel 3 es un mapeo débil.** Si la sección no tiene el tipo, la prenda se mapea a otra sección, pero
  siempre del mismo grupo de edad (adultos con adultos, niños con niños); si el grupo tampoco lo tiene,
  el artículo queda sin mapear. Las faldas de mujer (el catálogo solo trae faldas de niña y bebé) se
  mapean a vestidos de mujer por el ajuste de la tabla de arriba, en vez de caer en faldas de niña.
- **Los colores y tipos son aproximaciones.** `perceived_colour_master_name` agrupa tonos, y varios tipos
  de H&M (Top, Bodysuit, Coat, Underwear body…) no tienen un equivalente exacto.
- **El catálogo es el de hoy (oct 2026)**, también con items no disponibles; las compras son de 2020.
- **Precios:** solo se usan para ordenar; el nivel de precios de H&M y de Siman no es comparable.
"""
    os.makedirs(os.path.dirname(SAL_DOC), exist_ok=True)
    open(SAL_DOC, "w", encoding="utf-8").write(texto)


def main():
    cat = pd.read_csv(CATALOGO, dtype={"item_id": str})
    trans = pd.read_csv(TRANS_HM, dtype={"article_id": str, "customer_id": str})
    precios = trans.groupby("article_id")["price"].mean()
    art = preparar_articulos(set(precios.index), precios)
    assert len(art) == len(precios), "faltan artículos en articles.csv"

    # Artículos sin tipo o sin candidatos quedan con nivel 0
    art = asignar(art, cat)
    mapeo = art[art["nivel"] > 0][["article_id", "item_id", "nivel", "seccion", "tipo_prenda", "color_base"]]
    os.makedirs(os.path.dirname(SAL_MAPEO), exist_ok=True)
    mapeo.to_csv(SAL_MAPEO, index=False)

    tx = trans.merge(art[["article_id", "item_id", "nivel", "product_type_name", "tipo_prenda",
                          "seccion"]], on="article_id", how="left")
    tx = tx.merge(cat[["item_id", "precio_q", "seccion"]].rename(columns={"seccion": "seccion_siman"}),
                  on="item_id", how="left")
    ok = tx[tx["nivel"] > 0]
    ok[["t_dat", "customer_id", "item_id", "precio_q", "split"]].to_csv(SAL_TRANS, index=False)

    # --- Estadísticas ---
    n_tx, n_art = len(tx), len(art)
    niveles = {1: "1 (sección + tipo + color)", 2: "2 (sección + tipo)", 3: "3 (grupo de edad + tipo)"}
    por_nivel = pd.DataFrame({
        "artículos": art["nivel"].value_counts(),
        "% artículos": art["nivel"].value_counts() / n_art * 100,
        "transacciones": tx["nivel"].value_counts(),
        "% transacciones": tx["nivel"].value_counts() / n_tx * 100,
    }).reindex([1, 2, 3, 0]).fillna(0)
    por_nivel.index = [niveles.get(i, "sin mapear") for i in por_nivel.index]
    por_nivel.index.name = "nivel"
    por_nivel[["artículos", "transacciones"]] = por_nivel[["artículos", "transacciones"]].astype(int)
    por_nivel[["% artículos", "% transacciones"]] = por_nivel[["% artículos", "% transacciones"]].round(1)

    sin = tx[tx["nivel"] == 0]
    sin_tipo = sin[sin["tipo_prenda"].isna()]
    desc = sin_tipo.groupby("product_type_name").size().sort_values(ascending=False)
    descartes = pd.DataFrame({"transacciones": desc.head(8)})
    descartes.loc["otros tipos sin equivalente"] = desc.iloc[8:].sum()
    descartes.loc["tipo con equivalente pero sin candidatos"] = len(sin) - len(sin_tipo)
    descartes["% de las descartadas"] = (descartes["transacciones"] / max(len(sin), 1) * 100).round(1)
    descartes.index.name = "motivo (product_type_name)"

    por_item = ok["item_id"].value_counts()
    disp = cat.loc[cat["disponible"], "item_id"]
    n3 = ok[ok["nivel"] == 3].groupby(["seccion", "tipo_prenda", "seccion_siman"]).size()
    n3 = n3.sort_values(ascending=False).head(10).rename("transacciones").reset_index()
    n3["% de nivel 3"] = (n3["transacciones"] / (ok["nivel"] == 3).sum() * 100).round(1)
    n3.columns = ["sección H&M", "tipo", "sección del item Siman", "transacciones", "% de nivel 3"]
    n3.index = n3.index + 1
    n3.index.name = "#"
    kids = art[art["seccion"].isin(["nina", "nino"])]
    ninos_defecto = sum(pista_ninos(a, b, c) is None for a, b, c in zip(
        kids["section_name"], kids["department_name"], kids["prod_name"]))
    stats = {
        "clientes": trans["customer_id"].nunique(), "filas": n_tx, "articulos": n_art,
        "por_nivel": por_nivel, "descartes": descartes, "nivel3": n3,
        "desc_n": len(sin), "desc_pct": pct(len(sin), n_tx), "cons_pct": pct(len(ok), n_tx),
        "items_con_compra": len(por_item), "items_total": len(cat),
        "items_pct": pct(len(por_item), len(cat)),
        "items_disp": len(disp), "items_disp_con": int(disp.isin(por_item.index).sum()),
        "top10": pct(por_item.head(10).sum(), len(ok)), "top100": pct(por_item.head(100).sum(), len(ok)),
        "ninos_defecto": ninos_defecto,
        "sin_equivalente": sorted(set(art["product_type_name"]) - set(TIPO)),
    }
    escribir_doc(stats)

    print(f"Artículos: {n_art:,}   transacciones: {n_tx:,}")
    print("\nPor nivel:\n" + por_nivel.to_string())
    print(f"\nTransacciones descartadas: {len(sin):,} ({stats['desc_pct']})   conservadas: {stats['cons_pct']}")
    print(descartes.to_string())
    print(f"\nItems de Siman con al menos 1 compra: {len(por_item):,} de {len(cat):,} ({stats['items_pct']})")
    print(f"Concentración: top 10 items = {stats['top10']} de las compras, top 100 = {stats['top100']}")
    print("\nNivel 3, mayores:\n" + n3.to_string())
    print(f"\nNiños sin pista de género (→ {NINOS_POR_DEFECTO}): {ninos_defecto:,} artículos")
    print("\nTransacciones mapeadas por split:\n" + ok["split"].value_counts().to_string())

    # --- Comprobaciones ---
    assert ok["item_id"].isin(cat["item_id"]).all()
    n12 = ok[ok["nivel"].isin([1, 2])]
    assert (n12["seccion"] == n12["seccion_siman"]).all(), "nivel 1 o 2 con otra sección"
    nv3 = ok[ok["nivel"] == 3]
    assert (nv3["seccion"].map(GRUPO_EDAD) == nv3["seccion_siman"].map(GRUPO_EDAD)).all(), \
        "nivel 3 cruza adultos y niños"
    assert ok.loc[ok["split"] == "train", "t_dat"].max() < INICIO_VAL
    assert set(ok["split"]) == {"train", "val", "test"}
    assert len(ok) / n_tx > MIN_CONSERVADAS, f"solo se conserva {len(ok) / n_tx:.1%}"
    # El orden de precios se conserva: dentro de cada grupo, más caro en H&M -> no más barato en Siman
    x = art[art["nivel"] > 0].merge(cat[["item_id", "precio_q"]], on="item_id")
    for nivel, claves in NIVELES.items():
        for _, g in x[x["nivel"] == nivel].groupby(claves):
            assert g.sort_values(["precio", "article_id"])["precio_q"].is_monotonic_increasing
    print(f"\nGuardado en {SAL_MAPEO}, {SAL_TRANS} y {SAL_DOC}")


if __name__ == "__main__":
    main()
