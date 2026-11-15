# Plan de desarrollo — Proyecto Final CC3092

**Proyecto:** Recomendador de moda para e-commerce en Guatemala (catálogo de Siman GT)
**Equipo:** Javier Andre Benitez Garcia (23405) · Javier Alejandro López del Cid (23415) · Hugo Javier Cifuentes Carredano (23079)
**Horario de trabajo:** sesiones de 12:00 a 2:00 p. m., todos los días excepto martes
**Fechas clave:** presentaciones la semana del 16 de noviembre · entrega del trabajo escrito el **18 de noviembre de 2026**

---

## Resumen

| Fase | Fechas | Objetivo | Resultado principal |
|---|---|---|---|
| 1 | 24 – 30 oct | Tener los datos listos y una línea base | Catálogo limpio, muestra de H&M mapeada, EDA y versión A |
| 2 | 31 oct – 6 nov | Entrenar los modelos de deep learning | Versiones B y C entrenadas con métricas en test |
| 3 | 7 – 12 nov | Traducir métricas a dinero y desplegar | Modelo de valor + API y web funcionando |
| 4 | 13 – 15 nov | Documentar y preparar la presentación | Informe, README, video y ensayo |

---

## Estructura del repositorio

```
recomendador-moda-gt/
├── data/
│   ├── raw/            # datos originales (Siman, H&M) — no se suben a GitHub
│   ├── processed/      # datos limpios y mapeados
│   └── embeddings/     # vectores de texto e imagen
├── src/
│   ├── scraping/       # scraper_siman.py
│   ├── data/           # limpieza, muestreo y mapeo
│   ├── models/         # baseline, two-tower, multimodal
│   ├── evaluation/     # métricas y modelo de valor
│   └── api/            # FastAPI
├── web/                # página tipo tienda
├── notebooks/          # EDA y gráficas para el informe
├── reports/            # figuras, tablas y trabajo escrito
├── requirements.txt
└── README.md
```

---

## Fase 1 — Datos y línea base (24 – 30 oct)

**Objetivo:** tener todos los datos limpios, entendidos y una línea base contra la cual comparar.

### Datos
- Correr `scraper_siman.py` y ajustar el filtro de categorías de moda.
- Limpiar el catálogo: quitar duplicados, productos sin imagen o sin precio, y unificar colores y categorías.
- Descargar el dataset de H&M de Kaggle y tomar una muestra (por ejemplo, 6 meses y unos 50k clientes).
- **Mapeo H&M → Siman:** asociar cada artículo de H&M con prendas del catálogo local por categoría, tipo de prenda y color. Documentar las reglas usadas, porque es el supuesto principal del proyecto.
- Separar los datos por tiempo, igual que en el uso real: train (meses anteriores), validación (penúltima semana) y test (última semana).

### Análisis exploratorio (EDA)
- Distribución de categorías, colores, marcas y precios en quetzales.
- Compras por cliente y popularidad de productos (cola larga).
- Valores nulos, atípicos y posibles fugas de información entre train y test.

### Código
- `src/data/limpieza.py`, `src/data/muestreo.py`, `src/data/mapeo.py`
- `notebooks/01_eda.ipynb`
- `src/models/baseline.py`: **versión A**, recomienda los 12 productos más vendidos.
- `src/evaluation/metricas.py`: recall@12, precision@12 y MAP@12.

### Entregables de la fase
- [ ] `data/processed/catalogo.csv` y `data/processed/transacciones.csv`
- [ ] Gráficas del EDA guardadas en `reports/figuras/`
- [ ] Métricas de la versión A en test
- [ ] Documento corto con las reglas de mapeo

---

## Fase 2 — Modelos B y C (31 oct – 6 nov)

**Objetivo:** entrenar los dos modelos de deep learning y compararlos con la línea base.

### Datos
- Generar **embeddings de texto** de nombre + descripción + categoría con un modelo preentrenado (por ejemplo, un sentence-transformer multilingüe, porque los textos están en español).
- Generar **embeddings de imagen** con CLIP.
- Calcularlos una sola vez y guardarlos en `data/embeddings/` para no repetir el cómputo.

### Desarrollo de modelos
- **Versión B — two-tower:**
  - Torre de cliente: historial de compras reciente (promedio de los embeddings de sus prendas).
  - Torre de producto: embedding de texto + categoría + color + rango de precio.
  - Pérdida contrastiva con negativos dentro del batch.
- **Versión C — multimodal:** la versión B más el embedding de imagen en la torre de producto.
- **Cold start:** para clientes nuevos, recomendar las prendas más parecidas a una foto o a una prenda elegida (búsqueda por similitud de coseno).

### Entrenamiento
- PyTorch, optimizador Adam y early stopping según el recall@12 de validación.
- Probar pocos hiperparámetros: tamaño del embedding, learning rate y número de negativos.
- Registrar cada iteración: cambios, métricas en train/val/test, tiempo de entrenamiento y GPU usada.

### Código
- `src/models/embeddings.py`, `src/models/two_tower.py`, `src/models/multimodal.py`
- `src/models/train.py` y `src/models/recommend.py`

### Entregables de la fase
- [ ] Modelos B y C guardados (`.pt`)
- [ ] Tabla de iteraciones con métricas y costo de cómputo
- [ ] Curvas de entrenamiento de las iteraciones más importantes
- [ ] Ejemplos de recomendaciones para revisarlos visualmente

---

## Fase 3 — Modelo de valor y despliegue (7 – 12 nov)

**Objetivo:** demostrar con números cuánto dinero genera cada versión y dejar el sistema funcionando.

### Modelo de valor
- Función de valor:
  `Ventas extra al mes = visitas × CTR(recall@12) × tasa de conversión × ticket promedio (Q) − costo de servidor`
- Supuestos con fuente: tráfico, conversión y ticket promedio del e-commerce de moda en Guatemala (ICEX, Cámara de Comercio).
- **Curva de valor:** cómo cambian las ventas al subir el recall@12 o el número de recomendaciones mostradas.
- **Análisis de sensibilidad:** qué pasa si los supuestos están un 20 % o un 50 % equivocados.
- Comparación final: "C genera Q X más al mes que A, con un costo extra de Q Y".

### Despliegue
- **API (FastAPI):**
  - `GET /recomendar/{cliente_id}?version=A|B|C` devuelve las 12 prendas.
  - `POST /similares` recibe una foto o un ID de prenda (cold start).
- **Web tipo tienda:** muestra las recomendaciones de A, B y C lado a lado, con foto, precio y la razón de cada recomendación.
- Publicarlo en un servicio gratuito o barato (por ejemplo, Render o Hugging Face Spaces).
- Medir la latencia y el costo por predicción.

### Código
- `src/evaluation/modelo_valor.py` y `notebooks/02_modelo_valor.ipynb`
- `src/api/main.py`
- `web/index.html`

### Entregables de la fase
- [ ] Gráficas de la curva de valor y del análisis de sensibilidad
- [ ] Tabla final de valor por versión
- [ ] API y web publicadas, con la URL
- [ ] Métricas operativas: latencia, throughput y costo por predicción

---

## Fase 4 — Informe, video y ensayo (13 – 15 nov)

**Objetivo:** dejar todo listo para entregar y presentar.

### Trabajo escrito (secciones 4.1 a 4.9 del enunciado)
- Contexto de negocio y problema · datos y EDA · modelo de valor · metodología · resultados de iteraciones · despliegue · explicabilidad, ética y riesgos · conclusiones · enlace al repositorio.
- Explicabilidad: mostrar qué prendas del historial influyen en cada recomendación y revisar si el modelo funciona peor para alguna categoría o rango de precio.

### Repositorio
- README con instalación, cómo correr el scraper, cómo entrenar y cómo reproducir las métricas.
- `requirements.txt` con versiones fijas.

### Presentación
- Video de demostración de 2 a 4 minutos (`.mp4`).
- Diapositivas: pitch de 8 a 10 minutos y demo de 3 a 5 minutos.
- Preparar una entrada que no esté en el entrenamiento (por ejemplo, una foto de ropa nueva) para la demo.
- Ensayar las preguntas técnicas: **todos deben poder explicar cualquier parte del proyecto.**

### Entregables de la fase
- [ ] Trabajo escrito completo
- [ ] README y repositorio público
- [ ] Video `.mp4`
- [ ] Diapositivas y ensayo hecho

---

## Riesgos y cómo manejarlos

| Riesgo | Plan |
|---|---|
| El scraper trae pocas prendas o falla | Ajustar categorías; si no alcanza, usar el catálogo de H&M como respaldo |
| El mapeo H&M → Siman queda débil | Mapear por niveles (categoría → tipo → color) y documentarlo como supuesto |
| Falta de GPU o entrenamientos lentos | Embeddings precalculados una sola vez, muestra más pequeña y Google Colab |
| El tiempo no alcanza | Prioridad: A y B funcionando + modelo de valor; C es la mejora |
