---
title: Recomendador de Moda
emoji: 👗
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

Recomendador de moda para e-commerce en Guatemala (proyecto final de Deep Learning, UVG CC3092).

Modelo de dos torres que combina historial de compras, texto e imagen (CLIP) del catálogo.
Elige un cliente para ver sus recomendaciones (versiones A/B/C) o sube una foto para
encontrar prendas similares (cold start).
La API FastAPI y la web de demo corren en el mismo contenedor.
