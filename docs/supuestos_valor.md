# Supuestos del modelo de valor (Fase 3)

Fórmula: `ventas_v = visitas x f_ve x CTR_v x conv_click x ticket`, con `CTR_v = ctr_ref x recall_v / recall_A`.
`extra_v = (ventas_v - ventas_A) - (costo_v - costo_A)`. Código: `src/evaluation/modelo_valor.py`.
Fecha de consulta de todas las fuentes: 2026-11-05. Las cifras son órdenes de magnitud, no predicciones.

| Supuesto | Valor usado | Rango plausible | Fuente |
|---|---|---|---|
| Visitas/mes (`visitas`) | 321 000 | 226 000 - 600 000 | Semrush, tendencias retail El Salvador (sep-2026): https://www.semrush.com/trending-websites/sv/retail |
| Fracción que ve el carrusel (`f_ve`) | 50 % | 25 % - 75 % | Supuesto del equipo |
| CTR del carrusel con A (`ctr_ref`) | 2 % | 1 % - 4 % | Supuesto del equipo |
| Compra tras clic (`conv_click`) | 3.6 % | 0.8 % - 3.6 % | Salesforce Shopping Index vía DPL News (5-jul-2024): https://dplnews.com/?p=240714 (0.8 % LatAm) x 4.5 (clic en recomendación, Salesforce 2023) |
| Ticket (`ticket`) | Q896 | Q400 - Q1 100 | Datos del proyecto (`transacciones.csv`: 3.55 prendas x Q252 por cliente-día). Referencia externa: Q170 (2020) |
| Tipo de cambio (`tc`) | Q7.63 / USD | 7.50 - 7.80 | Banguat, citado por El Comercio (13-abr-2026): https://elcomercio.pe/mag/respuestas/us/convertir-dolar-a-quetzal-hoy-precio-del-usd-en-guatemala-para-este-13-de-abril-de-2026-nnda-nnrt-noticia |
| Costo CPU (`precio_hora_usd`) | US$0.03/h (730 h = US$21.9; Q167) | 0.03 - 0.06 | Hugging Face Spaces, CPU Upgrade 8 vCPU / 32 GB: https://huggingface.co/docs/hub/spaces-gpus |

## Detalle y citas

- **Visitas.** Cita: fila "siman.com 321.02K" en la tabla de tendencias de retail de El Salvador (sep-2026); la misma fuente dio 226.26K en feb-2026. Es el tráfico de El Salvador del dominio que comparte Siman; el de Guatemala no apareció con fuente (Similarweb no lo mostró). Se usa como orden de magnitud y el rango sube hasta 600 000 para cubrir Guatemala. Semrush son estimaciones, no datos propios de Siman.
- **f_ve y ctr_ref: sin fuente (supuesto del equipo).** No hallamos un benchmark independiente de CTR de carruseles. Solo anécdotas (un test de 2012 reportó 1.38 % y 6.54 % según el diseño; un blog dice que bajo 2 % es flojo). Se toma 2 % como valor medio conservador. Importa poco el nivel absoluto: el *extra* escala con él linealmente, la comparación entre versiones no.
- **conv_click.** Cita: "La tasa de conversión en América Latina fue del 0,8 %, comparada con el 0,9 % del primer trimestre de 2023" (Q1-2024, todas las categorías, no solo moda). El factor 4.5x (quien hace clic en una recomendación convierte 4.5 veces más) es el hallazgo del Shopping Index 2023, que solo vimos citado en un resultado de búsqueda, no lo abrimos en la fuente original. Por eso el rango baja hasta 0.8 %.
- **Ticket.** No encontramos un ticket online vigente de Guatemala con fuente. El 2.º Estudio de Comercio Electrónico de la Cámara de Comercio (Prensa Libre, 5-sep-2020) menciona "un ticket promedio de compra de Q170", pero es de 2020 y de otro tipo de comercio; no representa moda en Siman. Usamos el ticket real del dataset del proyecto (Q896, supuesto del equipo) y lo dejamos en el rango 400-1100. Si fuera Q170 el extra de todas las versiones se divide entre ~5.
- **Tipo de cambio.** Q7.63886 por USD (Banguat, 13-abr-2026, vía El Comercio) y 7.62929 (10-may-2026). No verificamos el dato de noviembre. Impacto mínimo.
- **Costo de servidor.** Tabla oficial de Hugging Face: CPU Basic gratis (2 vCPU, 16 GB); CPU Upgrade US$0.03/h. Facturado por minuto mientras corre; 730 h/mes.
  - A y la regla sin modelo: Q0 (lista fija o regla ligera, sin modelo).
  - B: 1 servidor CPU Upgrade (supuesto del equipo: 24/7).
  - C: 2 instancias (supuesto del equipo: CLIP exige más RAM/replica; HF factura cada réplica por separado).
