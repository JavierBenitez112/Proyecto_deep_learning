# Recomendador de moda para e-commerce en Guatemala

Proyecto final de **CC3092 – Deep Learning y Sistemas Inteligentes** (UVG, sección 20, Ing. Javier Fong).

**Equipo:** Javier Andre Benitez Garcia (23405) · Javier Alejandro López del Cid (23415) · Hugo Javier Cifuentes Carredano (23079)

## Idea

Sistema que recomienda las 12 prendas que un cliente tiene más probabilidad de comprar en una tienda departamental en línea de Guatemala (catálogo de **Siman GT**). Usa el historial de compras, el texto y las fotos de los productos.

Se comparan tres versiones:

| Versión | Qué usa |
|---|---|
| A – Línea base | Los productos más vendidos (sin deep learning) |
| B – Two-tower | Historial + texto + categorías |
| C – Multimodal | B + embeddings de imagen (CLIP) |

Cada versión se evalúa con recall@12 y se traduce a ventas extra al mes en quetzales con un modelo de valor.

## Datos

- **Catálogo local:** API pública VTEX de Siman Guatemala → `src/scraping/scraper_siman.py`
- **Patrón de compras:** [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations), mapeado al catálogo local
- **Modelo de valor:** cifras de e-commerce de Guatemala (ICEX, Cámara de Comercio)

## Fechas clave

- Desarrollo: 24 oct – 15 nov 2026 (plan en `docs/Plan_Fases_Proyecto_DL.md`)
- Presentaciones: semana del 16 nov
- **Entrega del trabajo escrito: 18 nov 2026**

## Estructura

```
docs/        enunciado, propuesta y plan de fases
src/         código (scraping, datos, modelos, evaluación, API)
data/        datos locales (no se suben a GitHub)
notebooks/   EDA y análisis
web/         página de demo
```

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/scraping/scraper_siman.py
```

El catálogo queda en `data/raw/catalogo_siman.csv` y las imágenes en `data/raw/imagenes/`.

## Despliegue

Google Cloud Run (proyecto `recomendador-moda-uvg`, región `us-central1`). Cloud Build construye el
`Dockerfile`; `.gcloudignore` define qué se sube (incluye los datos y modelos que `.gitignore` excluye).

```bash
gcloud run deploy recomendador-de-moda --source . --project recomendador-moda-uvg --region us-central1 --memory 2Gi --cpu 1 --port 7860 --min-instances 0 --max-instances 1 --cpu-boost --no-allow-unauthenticated
```

El servicio es **privado** (sin token responde 403). Para abrirlo en el navegador:

```bash
gcloud run services proxy recomendador-de-moda --region us-central1 --project recomendador-moda-uvg
```

y abrir http://localhost:8080. Para hacerlo público:

```bash
gcloud run services add-iam-policy-binding recomendador-de-moda --region us-central1 --project recomendador-moda-uvg --member=allUsers --role=roles/run.invoker
```

URL: https://recomendador-de-moda-597180544725.us-central1.run.app
