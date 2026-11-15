# Transfer learning: comparación contra el modelo base

Sección para el trabajo escrito (metodología y resultados). Responde al requisito de documentar el cambio de desempeño al aplicar transfer learning frente al modelo base.

## Qué se reutiliza y qué se entrena

El proyecto no usa ningún LLM. El modelo es un recomendador two-tower diseñado y entrenado por el equipo (`src/models/two_tower.py`). La técnica de transfer learning que aplica es **extracción de características**: dos modelos preentrenados se usan congelados para describir cada prenda, y sobre esos vectores se entrena nuestra red.

| Componente | Origen | ¿Se entrena? |
|---|---|---|
| Embedding de texto (768 dims) | `intfloat/multilingual-e5-base` | No, congelado |
| Embedding de imagen (512 dims, solo versión C) | `openai/clip-vit-base-patch32` | No, congelado |
| Embeddings de sección, tipo de prenda, color y rango de precio | Propio | Sí |
| Embedding de id de item | Propio | Sí |
| Torre de producto y torre de cliente (MLP) | Propio | Sí |

La pérdida es un softmax contrastivo con negativos dentro del batch y corrección logQ.

## Modelo base y variantes comparadas

- **Modelo base (sin entrenar):** los embeddings preentrenados tal cual. El vector de un cliente es el promedio, ponderado por recencia, de los embeddings de sus últimas 30 compras; se recomiendan las 12 prendas con mayor coseno. No hay torres ni entrenamiento.
- **Two-tower sin preentrenados:** nuestra red entrenada solo con categorías e id de item, sin e5 ni CLIP. Mide cuánto aportan los embeddings preentrenados.
- **Modelo completo:** nuestra red entrenada sobre los embeddings preentrenados más las categorías y el id.

Cada variante se reporta de dos formas: **puro** (solo el coseno del modelo) y **combinado** (coseno + popularidad de la semana previa + regla de recompra, con pesos elegidos en validación). Métrica: recall@12 en la semana de test (4,592 clientes).

## Resultados

Recall@12 en test. Entre paréntesis, el cambio frente al modelo base de la misma columna.

| Variante | B puro | B combinado | C puro | C combinado |
|---|---|---|---|---|
| Modelo base (sin entrenar) | 0.0421 | 0.1043 | 0.0401 | 0.1035 |
| Two-tower sin preentrenados | 0.0624 (+0.0202) | 0.1061 (+0.0017) | — | — |
| Modelo completo, misma configuración | 0.0629 (+0.0208) | 0.1052 (+0.0008) | 0.0596 (+0.0195) | 0.1064 (+0.0029) |

B usa texto; C usa texto + imagen. La fila "modelo completo" usa en ambas versiones la configuración ganadora de B (d = 128, lr = 1e-3, batch = 512, logQ, 12 semanas de entrenamiento, id de item), para que la única diferencia entre B y C sea la imagen.

Referencias sin deep learning: popularidad (versión A) 0.0743; popularidad + recompra sin modelo 0.1035.

Tablas completas (validación, precision@12 y MAP@12): `reports/resultados/transfer_B.csv` y `transfer_C.csv`.

## Interpretación

1. **El entrenamiento mejora al modelo base.** En puro, el recall@12 sube de 0.0421 a 0.0629 en B (+49 %) y de 0.0401 a 0.0596 en C (+49 %). Los embeddings preentrenados capturan parecido entre prendas; el entrenamiento les enseña qué se compra junto.
2. **Los embeddings preentrenados aportan poco sobre las categorías y el id.** En B, el two-tower sin ellos llega a 0.0624 frente a 0.0629 del completo. La mejora viene del entrenamiento con las transacciones, no de las características heredadas.
3. **La imagen no mejora de forma medible.** Con la misma configuración, C queda 0.0034 por debajo de B en puro y 0.0012 por encima en combinado (en validación, 0.0003 por debajo). No hay una señal consistente.
4. **Combinado con las reglas, todo converge.** Todas las variantes combinadas quedan entre 0.1035 y 0.1064, y la regla sin modelo ya alcanza 0.1035. El aporte del modelo sobre las reglas es de 0.001 a 0.003.

## Nota sobre la versión C guardada

El checkpoint `data/modelos/version_C.pt` se eligió por recall combinado en validación y ganó con otra configuración (d = 64, lr = 3e-4, sin logQ, 4 semanas, sin id de item). Sin la corrección logQ, su recall puro es 0.0214, por debajo de su modelo base (0.0401), aunque combinado llega a 0.1036. Esa caída se debe a la configuración y no a la imagen: el two-tower sin preentrenados con esa misma configuración da 0.0206.

## Limitaciones

- **Una sola semilla.** Entre dos corridas idénticas el recall varió unas 0.0003 (entrenamiento en MPS, no determinista). Las diferencias de 0.001 a 0.003 en la columna combinada no deben leerse como reales.
- **Transacciones mapeadas.** El patrón de compras viene de H&M y se mapeó al catálogo de Siman por categoría, tipo y color. Esto puede borrar parte de la información fina que los embeddings de texto e imagen podrían aprovechar.
- **Sin fine-tuning.** Los modelos preentrenados quedaron congelados; no se ajustaron sus pesos.

## Reproducción

```bash
python src/models/two_tower.py --transfer            # versión B
python src/models/two_tower.py --transfer --imagen   # versión C
```

Cada comando carga el checkpoint guardado, verifica que reproduce su recall@12 de test y escribe `reports/resultados/transfer_{B,C}.csv`. No modifica los modelos ni los demás resultados.
