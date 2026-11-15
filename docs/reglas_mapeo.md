# Reglas del mapeo H&M → Siman

Las compras del proyecto son las de H&M (últimos 6 meses, 50,000 clientes), pero el
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
| Baby/Children (resto) y Kids Sports | nina / nino: palabra Girl/Boy en section_name, luego en department_name, luego en prod_name. Sin pista (47 artículos) → **nina** |
| Divided, Sport | hombre si section_name dice Men; si no, mujer |

**Tipo** (`product_type_name`, unos 130 valores, → tipo de Siman):

| tipo_prenda (Siman) | product_type_name (H&M) |
|---|---|
| accesorio cabello | Alice band, Hair clip, Hair string, Hair ties, Hair/alice band, Hairband, Headband |
| anillo | Ring |
| arete | Earring, Earrings |
| babero | Baby Bib |
| billetera | Wallet |
| blazer | Blazer |
| blusa | Blouse, Top |
| bolso | Bag, Backpack, Bumbag, Cross-body bag, Shoulder bag, Tote bag, Weekend/Gym bag |
| bota | Boots, Bootie |
| brasier | Bra, Underwear body, Underwear set |
| bufanda | Scarf |
| calcetin | Socks, Leg warmers |
| camisa | Shirt |
| camiseta | T-shirt, Vest top, Bodysuit |
| camison | Night gown, Robe |
| chaleco | Outdoor Waistcoat, Tailored Waistcoat |
| chaqueta | Jacket, Coat |
| cinturon | Belt |
| collar | Necklace |
| complemento formal | Braces |
| conjunto | Garment Set |
| corbata | Tie |
| faja | Underdress |
| falda | Skirt |
| gorra | Beanie, Bucket hat, Cap, Cap/peaked, Felt hat, Hat/beanie, Hat/brim, Straw hat |
| guante | Gloves |
| jumpsuit | Jumpsuit/Playsuit |
| legging | Leggings/Tights |
| lentes de sol | Sunglasses |
| overol | Dungarees, Outdoor overall |
| pantalon | Trousers, Outdoor trousers |
| panti | Underwear bottom, Underwear Tights |
| pantufla | Slippers |
| pijama | Pyjama bottom, Pyjama jumpsuit/playsuit, Pyjama set, Sleeping sack, Sleep Bag |
| polo | Polo shirt |
| pulsera | Bracelet |
| reloj | Watch |
| ropa interior | Kids Underwear top, Long John |
| sandalia | Sandals, Heeled sandals, Flip flop, Wedge |
| short | Shorts |
| sombrilla | Umbrella |
| sudadera | Hoodie |
| sueter | Sweater, Cardigan |
| tacon | Heels, Pumps |
| tenis | Sneakers |
| traje de bano | Bikini top, Swimsuit, Swimwear bottom, Swimwear set, Swimwear top |
| vestido | Dress |
| zapato | Ballerinas, Flat shoe, Flat shoes, Moccasins, Other shoe, Pre-walkers |

Sin equivalente (se descartan): Accessories set, Bra extender, Clothing mist, Costumes, Dog Wear, Dog wear, Eyeglasses, Fine cosmetics, Giftbox, Keychain, Marker pen, Mobile case, Nipple covers, Other accessories, Sarong, Sewing kit, Side table, Soft Toys, Stain remover spray, Unknown, Washing bag, Waterbottle, Wireless earphone case, Zipper head.

Además, `Trousers` con apariencia Denim (o sección Denim) → **jeans**. Y estos ajustes por sección,
porque Siman clasifica distinto según la sección:

| tipo traducido | seccion | tipo final |
|---|---|---|
| blusa | hombre | camiseta |
| bota | bebe | zapato |
| bota | nina | zapato |
| bota | nino | zapato |
| camisa | mujer | blusa |
| camisa | nina | blusa |
| camiseta | nina | blusa |
| chaqueta | bebe | sueter |
| chaqueta | hombre | sudadera |
| chaqueta | nina | sueter |
| chaqueta | nino | sueter |
| falda | mujer | vestido |
| jumpsuit | bebe | mameluco |
| jumpsuit | nina | vestido |
| legging | bebe | pantalon |
| legging | nina | pantalon |
| panti | hombre | boxer |
| panti | nino | boxer |
| polo | bebe | camiseta |
| polo | mujer | camiseta |
| polo | nina | blusa |
| polo | nino | camiseta |
| sandalia | bebe | zapato |
| sandalia | nina | zapato |
| sandalia | nino | zapato |
| sudadera | bebe | sueter |
| sudadera | mujer | sueter |
| sudadera | nina | sueter |
| sudadera | nino | sueter |
| tenis | bebe | zapato |
| tenis | nina | zapato |
| tenis | nino | zapato |

**Color** (`perceived_colour_master_name` → `color_base`). Si `graphical_appearance_name` es
All over pattern, Argyle, Check, Colour blocking, Dot, Mixed solid/pattern, Other pattern, Stripe, el color es **multicolor**.

| color_base | perceived_colour_master_name |
|---|---|
| amarillo | Yellow |
| azul | Blue, Turquoise |
| beige | Beige, Mole |
| blanco | White |
| cafe | Brown |
| gris | Grey |
| metalico | Metal |
| morado | Lilac Purple |
| naranja | Orange |
| negro | Black |
| otro | Unknown, undefined |
| rojo | Red |
| rosado | Pink |
| verde | Green, Khaki green, Yellowish Green, Bluish Green |

## Estadísticas

Muestra: 539,807 transacciones y 31,166 artículos de H&M.

**Por nivel de mapeo**

| nivel | artículos | % artículos | transacciones | % transacciones |
| --- | --- | --- | --- | --- |
| 1 (sección + tipo + color) | 28,171 | 90.4 | 510,773 | 94.6 |
| 2 (sección + tipo) | 2,344 | 7.5 | 21,239 | 3.9 |
| 3 (grupo de edad + tipo) | 98 | 0.3 | 200 | 0.0 |
| sin mapear | 553 | 1.8 | 7,595 | 1.4 |

**Transacciones descartadas:** 7,595 (1.4%). Conservadas: 98.6%.

| motivo (product_type_name) | transacciones | % de las descartadas |
| --- | --- | --- |
| Unknown | 4,145 | 54.6 |
| Other accessories | 1,323 | 17.4 |
| Sarong | 921 | 12.1 |
| Costumes | 174 | 2.3 |
| Nipple covers | 96 | 1.3 |
| Mobile case | 86 | 1.1 |
| Bra extender | 35 | 0.5 |
| Dog wear | 17 | 0.2 |
| otros tipos sin equivalente | 76 | 1.0 |
| tipo con equivalente pero sin candidatos | 722 | 9.5 |

**Items de Siman que reciben al menos 1 compra:** 8,614 de 11,786
(73.1%); de los 7,650 disponibles, 5,486.

**Concentración:** los 10 items más comprados se llevan el 8.2% de las compras y los
100 más comprados el 23.9%.

**Compras por sección y tipo con más transacciones en nivel 3** (la sección de H&M no tiene ese tipo en
Siman, así que la prenda cae en otra sección del mismo grupo de edad):

| # | sección H&M | tipo | sección del item Siman | transacciones | % de nivel 3 |
| --- | --- | --- | --- | --- | --- |
| 1 | nina | ropa interior | bebe | 75 | 37.5 |
| 2 | bebe | traje de bano | nina | 24 | 12.0 |
| 3 | nina | overol | bebe | 18 | 9.0 |
| 4 | bebe | lentes de sol | nina | 15 | 7.5 |
| 5 | nino | blusa | nina | 14 | 7.0 |
| 6 | hombre | complemento formal | unisex | 12 | 6.0 |
| 7 | bebe | traje de bano | nino | 7 | 3.5 |
| 8 | bebe | accesorio cabello | nina | 5 | 2.5 |
| 9 | nino | overol | bebe | 5 | 2.5 |
| 10 | nino | guante | bebe | 5 | 2.5 |

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
