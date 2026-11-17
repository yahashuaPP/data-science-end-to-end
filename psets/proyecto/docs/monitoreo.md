# Monitoreo del modelo de GrantMatch

El servicio ([`servicio/api.py`](../servicio/api.py)) mide cada predicción con un middleware, resume esas mediciones en `/metricas`, y guarda cada predicción en `monitoreo/predicciones.jsonl`. Cada línea tiene:
- el identificador de la predicción y el momento;
- la versión del modelo;
- la entrada completa, con `award_key` y `work_key`;
- la probabilidad y la decisión.

[`monitoreo/desempeno.py`](../monitoreo/desempeno.py) une ese registro con la verdad y calcula las métricas por semana y por versión.

**La verdad llega muy tarde.** Un par (award, investigador) se confirma cuando una publicación del investigador aparece en OpenAlex reconociendo ese award, y eso tarda meses o años: la publicación sale después del financiamiento y OpenAlex la enlaza después.

Por eso hay dos tipos de métricas de modelo:
- **Las que se conocen el mismo día:** tasa de recomendación, probabilidad promedio y calidad de la entrada. No dicen si el modelo acierta, solo si recibe y produce lo mismo que en la validación.
- **Las que se conocen cuando llega la verdad:** AUC y recall. Solo se calculan para las semanas "maduras", a las que ya pasaron 365 días desde su último día. Antes, un par sin confirmar todavía puede confirmarse.

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Umbral de alerta |
|---|---|---|---|---|
| Peticiones a `/predecir` | Sistema | `/metricas` (middleware) | Al instante | Ninguna petición en una noche en que corre el lote (el lote no llamó al servicio) |
| Errores de validación (422) | Sistema | `/metricas` | Al instante | Más del 2 % de las peticiones. Con datos de la OBT se rechaza alrededor del 0.3 %, los pares sin campo de la publicación |
| Errores de servidor (≥ 500) | Sistema | `/metricas` | Al instante | Cualquiera: el servicio no debería fallar con una entrada válida |
| Latencia p50 y p95 | Sistema | `/metricas` | Al instante | p95 sobre 100 ms. Medida en 400 pares: p50 8 ms, p95 12 ms |
| Tasa de recomendación (fracción con `recomendar` = true) | Modelo, indirecta | Registro de predicciones | El mismo día | Fuera de ±25 % de la de la validación por award (al umbral de 0.10) |
| Probabilidad promedio | Modelo, indirecta (deriva de las predicciones) | Registro de predicciones | El mismo día | Cambio de más de 0.1 respecto a la semana de referencia, o PSI > 0.2 de su distribución |
| Pares con el campo del award / país del financiador | Datos (calidad de la entrada) | Registro de predicciones | El mismo día | Caída a la mitad respecto a la OBT (hoy 2.5 % y 95 %): el pipeline dejó de traer esas columnas |
| Pares en que el país del financiador coincide con el de un autor | Datos (deriva de la feature que más pesa) | Registro, con `agregar_features` | El mismo día | Fuera de ±25 % de su valor en entrenamiento |
| AUC por semana y versión | Modelo | Registro unido con `monitoreo/verdad.csv` (pares nuevos de `GOLD.FACT_AWARD_WORKS`) | 365 días después de la semana | Bajo 0.85 (validación por award: 0.897 ± 0.005) |
| Recall al umbral de 0.10 | Modelo (la métrica del negocio) | Igual que la AUC | 365 días después de la semana | Bajo 0.90 (validación: 0.982) |

**Qué se hace con una alerta:**
- **Alerta de sistema:** se revisa el servicio, los logs y el contenedor.
- **Alerta indirecta o de datos:** se revisa primero el pipeline (¿cambió la OBT?) y después los datos de entrada. Si la deriva es real, se reentrena con datos recientes y la versión nueva entra con el criterio de [`despliegue.md`](despliegue.md#versionado-y-registro).
- **Caída de AUC o de recall:** confirma que el modelo empeoró. Se reentrena y se compara contra la versión en producción.

## Deriva

**Cómo se simuló.** [`monitoreo/simular_deriva.py`](../monitoreo/simular_deriva.py) toma ventanas de 500 pares de prueba, lo que el servicio puntúa en una noche, y mide en cada una el PSI de tres features y la AUC del modelo final:
- `funder_en_pais_autor`, la que más pesa;
- `funder_country_code`;
- `cited_by_count`.

La referencia es el período de prueba (awards desde 2020), no el entrenamiento. La prueba entera ya está lejos del entrenamiento: PSI 0.27 en `funder_country_code`, porque los financiadores de China bajan de 17 % a 7.5 % de los pares, y 0.14 en `cited_by_count`, porque las publicaciones nuevas tienen menos citas. Aun así, el modelo da AUC 0.90 en ese período.

Eso ya es una lección: hubo deriva de datos real entre períodos y el modelo no empeoró. Por eso la alerta compara contra el período más reciente en que el modelo se validó. Esa referencia queda en `monitoreo/referencia_deriva.json`, que el servicio carga al arrancar.

**Las dos derivas simuladas:**
- **De datos:** una temporada con muchas más convocatorias de China. Los pares con financiador `CN` salen `peso` veces más.
- **Conceptual:** los financiadores empiezan a financiar a autores de otros países (programas internacionales). Los pares son los mismos, pero en una fracción de los pares reales con `funder_en_pais_autor` = 1 el financiamiento ya no ocurre, y en la misma cantidad de pares con 0 sí.

| Escenario (ventanas de 500 pares) | PSI `funder_en_pais_autor` | PSI `funder_country_code` | PSI `cited_by_count` | AUC |
|---|---|---|---|---|
| Normales (máximo de 200) | 0.025 | 0.055 | 0.042 | 0.94 (promedio 0.90) |
| Datos: China ×4 | 0.004 | 0.250 | 0.014 | 0.895 |
| Datos: China ×8 | 0.011 | 0.696 | 0.022 | 0.886 |
| Conceptual: 25 % internacional | 0.001 | 0.021 | 0.016 | 0.680 |
| Conceptual: 50 % internacional | 0.001 | 0.021 | 0.016 | 0.460 |

**Umbral elegido: PSI > 0.10 en cualquiera de las tres features, con la ventana llena (500 pares).** En 200 ventanas normales el PSI más alto fue 0.055, así que 0.10 no da ninguna falsa alarma (con 0.05 daba 0.5 %). En 200 ventanas de China ×4 detecta el 100 %: su PSI promedio de país es 0.25. El servicio lo responde en `/deriva`.

**Qué no detecta esta alerta:**
- **La deriva conceptual.** Con 50 % de financiamiento internacional la AUC cae de 0.90 a 0.46, peor que el azar, y el PSI no se mueve, porque los pares que llegan son los mismos. Solo la ve la AUC o el recall de `monitoreo/desempeno.py`, cuando llega la verdad, un año después.
- **Lo que pasa fuera de las tres features vigiladas:** un cambio en el tipo de financiamiento o en los dominios, por ejemplo.

Por otro lado, la alerta sí se enciende con derivas de datos que casi no dañan al modelo: con China ×8, la AUC solo baja de 0.90 a 0.89. Una alerta de deriva dice "revisar", no "reentrenar".
