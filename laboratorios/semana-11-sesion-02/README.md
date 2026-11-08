# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Que una exactitud alta no garantiza que el modelo sirva, sobre todo con clases desbalanceadas. El ejemplo es una enfermedad que afecta a 1 de cada 100 personas: un modelo que dice "sano" a todos acierta el 99 % y no detecta al único enfermo, que era lo que tenía que hacer. En su lugar recomienda métricas que miran la clase minoritaria: precision, recall y F1 (y PR-AUC en el caso de los LLM).

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Da dos ejemplos de negocio: a un equipo de medios le pidieron subir 20 % la duración de las sesiones, y alargaron las introducciones con relleno, que bajó la calidad de los videos; y una app de suscripciones escondió el botón de cancelar, así que el churn bajó porque la gente no encontraba la salida, no porque estuviera más contenta. Con modelos, el ejemplo es optimizar un LLM para ROUGE: aprende a copiar frases del texto y saca buen puntaje con resúmenes peores. Elegir un modelo por una sola métrica puede premiar al que se ajusta a esa métrica y no al que es más útil, por eso hay que mirar varias métricas y el criterio humano.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Bastante para lo que separa a los modelos. El mismo gradient boosting dio 1.2754 con la división de semilla 42 y 1.1867 con la de semilla 7: casi 0.09 de diferencia solo por cambiar qué viajes caen en prueba, más que lo que separaba al gradient boosting del bosque (1.2754 contra 1.3016). En el registro se ve en la tabla del Ejercicio 1 de la Semana 11.1: las dos corridas de gradient boosting tienen la misma huella de datos y los mismos hiperparámetros, y solo cambia `semilla_division`. Con validación cruzada al azar, los cinco pliegues del gradient boosting van de 1.16 a 1.28.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Esperaba que fueran parecidos pero no iguales, con unas centésimas de diferencia: cada pliegue tiene 595 viajes distintos, algunos más difíciles. Así fue: de 1.16 a 1.28, con promedio 1.2288 y desviación 0.036. El bosque, de 1.25 a 1.34 (promedio 1.2880).

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor. Los primeros pliegues entrenan con pocos viajes (el primero, con una sexta parte), y además el modelo tiene que estimar viajes de fechas que no vio, si algo cambió con el tiempo. Al azar, entrenamiento y validación se mezclan en las mismas fechas. Salió 1.3099 de promedio contra 1.2288, con más variación (desviación 0.069).

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El que deja fuera el aeropuerto: si JFK Airport no aparece como origen en el entrenamiento, el modelo no aprende bien la tarifa fija de 70 desde JFK. Así fue: el pliegue con Brooklyn y JFK Airport dio 6.43, contra 1.19 a 1.31 de los otros cuatro.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20 %: cuenta directamente los viajes en que el pasajero reclamaría. El MAE (1.28) y el RMSE (1.92) mezclan errores hacia arriba y hacia abajo, y en dólares, sin importar si el viaje era de 10 o de 70. En prueba, 26 de 595 viajes (4.4 %) se subestiman en más de 20 %.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema | Promedio | Desviación | Peor pliegue |
|---|---|---|---|
| Al azar, gradient boosting | 1.2288 | 0.0361 | 1.28 |
| Al azar, bosque | 1.2880 | 0.0291 | 1.34 |
| Temporal, gradient boosting | 1.3099 | 0.0693 | 1.40 |
| Por zona, gradient boosting | 2.2996 | 2.0648 | 6.43 (Brooklyn y JFK Airport) |

Métricas sobre los 595 viajes de prueba: MAE 1.2754, RMSE 1.9213, 4.4 % de viajes subestimados en más de 20 %.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy 0.9508 y recall 0: acierta en el 95 % de los pagos porque el 95 % son legítimos, y no detecta ni un fraude.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy sube de 0.9508 a 0.9712, apenas 2 puntos. El recall sube de 0 a 0.461: detecta casi la mitad de los fraudes. El recall describe mejor lo que gana la empresa: cada fraude detectado es un contracargo que no paga, y la accuracy casi no se mueve porque los fraudes son solo el 5 % de los pagos.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con 0.5 deja pasar 159 de los 295 fraudes (detecta 136), con 14 falsas alarmas: costo de 6402 dólares. El paso 5 eligió el umbral 0.1: deja pasar 122 fraudes y marca 222 falsas alarmas, con costo de 5546 dólares. Como un fraude cuesta 40 y una falsa alarma 3, conviene marcar más pagos aunque muchos sean legítimos.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

Sube. Con las falsas alarmas a 20 dólares, marcar de más deja de convenir: el costo de cada umbral queda en 15 760 (0.05), 9320 (0.1), 7820 (0.15), 7300 (0.2), 6780 (0.3) y 6640 (0.5), y el de menor costo pasa a ser 0.5.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

Por grupo, con `award_key` como grupo (`GroupKFold`), en [`validacion_proyecto.ipynb`](../../psets/proyecto/modelado/validacion_proyecto.ipynb). El mismo award aparece en muchos pares: todas las publicaciones que financió, más sus pares negativos. Al azar, un award puede caer a la vez en entrenamiento y en validación, y el modelo se evalúa con convocatorias que ya conoce. Por grupo, cada award queda entero de un lado, como pasa en producción con una convocatoria nueva. La división por fecha de la Semana 9.2 cumple lo mismo para una sola división; para validación cruzada, por grupo da más pliegues comparables.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

⟨Completar con la tabla de `validacion_proyecto.csv`.⟩ Por award, el modelo final da AUC ⟨X⟩ ± ⟨X⟩; al azar, ⟨X⟩ ± ⟨X⟩. ⟨Si al azar sale más alta: el esquema al azar sobreestima porque el modelo ve el mismo award en entrenamiento y en validación. Si salen casi iguales: con negativos al azar, las features que comparan los dos lados (`mismo_campo`, `funder_en_pais_autor`) no dependen de qué award es, así que mezclar awards no filtra información.⟩

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

⟨Completar.⟩ El final (⟨modelo⟩) y el segundo (⟨modelo⟩) difieren en ⟨X⟩ de AUC por award, y la desviación entre pliegues es ⟨X⟩. ⟨Si la diferencia es menor que la desviación: no se distinguen con estos datos, y me quedo con el más simple y rápido de los dos. Si es mayor: me quedo con el final.⟩

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

El costo esperado con un umbral de decisión, y el recall. En GrantMatch el error caro es el falso negativo: no recomendar un award que sí era para el investigador, que puede perder un financiamiento. El falso positivo le cuesta unos minutos de leer y descartar una convocatoria. Lo aproximé como 10 a 1. La AUC dice qué tan bien ordena el modelo los pares, pero no cuántos awards pertinentes se pierden con el umbral que se usa ni cuánto cuesta. Con 0.5, el costo es ⟨X⟩ y el recall ⟨X⟩; con el umbral de menor costo, ⟨X⟩, el costo baja a ⟨X⟩ y el recall sube a ⟨X⟩.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa se podría calcular por lotes: las combinaciones de zona de origen, zona de destino, hora y día son pocas (8 × 8 × 24 × 7), y se puede guardar una tabla cada noche y consultarla al pedir el viaje, ajustando por la distancia estimada. El fraude tiene que responder en el momento: depende del pago concreto (el monto, si la tarjeta es nueva, los intentos fallidos), que no existe hasta que el pasajero paga, y la respuesta tiene que llegar antes de aceptar el pago, en menos de un segundo.
