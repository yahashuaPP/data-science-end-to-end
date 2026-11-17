# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

Es el proceso de punta a punta con el que una organización establece, aplica y mantiene controles sobre cómo se usan sus modelos, para que funcionen como se espera, cumplan las normas y den resultados confiables con el tiempo. Nació en finanzas y ahora se aplica también a IA y aprendizaje automático. El artículo la forma con cinco componentes: (1) desarrollo y documentación del modelo (objetivo, datos de entrenamiento, metodología, supuestos y salidas esperadas); (2) un inventario central de modelos; (3) validación y back-testing (probar con datos históricos, sensibilidad, y en ML sesgo, robustez y sobreajuste); (4) despliegue y monitoreo continuo del desempeño y la deriva, con los controles metidos en el pipeline de MLOps / CI/CD; y (5) roles, responsabilidades y flujos de trabajo, con responsables claros entre científicos de datos, riesgo, negocio, cumplimiento y auditoría.

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Para cada modelo: para qué sirve, quién es su dueño, con qué metodología se construyó y en qué etapa de su ciclo de vida está (en desarrollo, en producción, retirado), incluidos los modelos escondidos en hojas de cálculo. A eso yo le sumaría lo que el registro de MLflow guarda solo: la versión, la corrida que la produjo (datos, hiperparámetros y métricas) y qué versión está en producción. Quien no lo entrenó lo necesita para saber a quién preguntar, si el modelo sigue vigente, con qué datos y contra qué se validó, y así evaluar su riesgo y decidir si lo usa, lo cambia o lo retira sin tener que reconstruir la historia.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Con un solo archivo que se sobrescribe, no lo sabría: el `modelo.joblib` de hoy ya no existe mañana. Hace falta que cada predicción diga qué modelo la produjo (una versión o un `run_id` en la respuesta y en el registro de predicciones) y que los modelos anteriores no se borren. Con un registro de modelos cada entrenamiento es una versión numerada que apunta a su corrida (datos, parámetros y métricas): con la versión que quedó en la predicción encuentro el modelo exacto. Para volver atrás, en vez de reentrenar, muevo el alias de producción a la versión anterior, vuelvo a exportarla y reinicio el servicio.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

`promover.py` termina con `RestException: INVALID_PARAMETER_VALUE: Registered model alias produccion not found`. Dice que en el registro no hay nada con ese alias, porque no hay nada en absoluto: `registrar_modelo.py` entrenó la red y guardó su MAE (1.1695) en una corrida, pero sin el `TODO 1` nunca registró el modelo, así que no existe el modelo `tarifa-taxis` ni ninguna versión (por eso `Version None` y `versiones.py` no imprime nada). Una corrida con métricas no es lo mismo que un modelo registrado.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Dos versiones: la 1 (64 neuronas, MAE 1.1695) y la 2 (8 neuronas, MAE 1.1877). A la primera no le pasó nada: registrar la segunda no la reemplaza ni la borra, solo agrega una versión con el número siguiente, cada una con su corrida y su métrica. Ninguna tiene alias todavía.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

Con `produccion` en la versión 2, `/salud` dice `"version_modelo": "2"`. Al mover el alias a la versión 1 sin reiniciar, el servicio sigue respondiendo con la versión 2: el modelo se carga una sola vez, al arrancar, y queda en memoria. Recién después de reiniciar uvicorn `/salud` dice `"1"`. Mover el alias no cambia nada en un servicio que ya está corriendo: hay que reiniciarlo (o redesplegarlo).

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Falla `test_cada_prediccion_dice_que_version_la_calculo`: la respuesta de `/predecir` solo tenía la tarifa, sin `version_modelo`. Comprueba que cada predicción lleva la versión que la calculó, mientras que `/salud` solo dice qué versión está cargada ahora. Si entre una petición y otra se reinicia el servicio con otro modelo, `/salud` ya no dice nada sobre las predicciones de antes: la versión tiene que viajar con cada tarifa para poder rastrearla después.

**Lo que respondió el servicio:**

Viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta): con la versión 2 (8 neuronas), `{"tarifa_estimada": 69.98, "version_modelo": "2"}`; después de mover el alias a la 1 sin reiniciar, la misma respuesta con `"2"`; y después de reiniciar, con la versión 1 (64 neuronas), `{"tarifa_estimada": 70.07, "version_modelo": "1"}`. `pytest -v`: 2 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No hubo votación: en esta sesión no se hizo el ConcepTest en clase.

**Tu razonamiento:**

No aplica (sin ConcepTest en clase).

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica (sin ConcepTest en clase).

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedó en producción la versión 2, el gradient boosting (recall 0.661, precisión 0.429 al umbral de 0.10). La versión 1, la logística (recall 0.644), entró primero porque no había ninguna; el gradient boosting la superó en recall y se llevó el alias. La última, la 3 (logística muy regularizada), tiene recall 0.102: detecta muchos menos fraudes, así que la regla la rechazó. Registrar una versión no la pone en producción: solo entra si mejora a la que está.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla compara solo el recall, y la logística con C=0.001 detecta apenas el 10.2 % de los fraudes: su precisión es alta (0.667) porque casi no da alarmas, pero deja pasar casi todo. Si las falsas alarmas costaran más que los fraudes no detectados, la regla tendría que comparar la precisión, o mejor, el costo esperado con los costos de cada error (por ejemplo, costo = falsos negativos × costo de un fraude + falsos positivos × costo de una revisión), y promover la versión con menor costo. Probablemente también habría que subir el umbral de 0.10.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En el registro mismo: en la versión 3 de `fraude-pagos` (pestaña Models), sus etiquetas dicen `validacion` = `rechazada` y `comparada_con` = `2`, y su corrida (enlazada desde la versión) guarda las métricas: recall 0.102 contra 0.661 de la versión 2. Con eso cualquiera reconstruye la decisión sin preguntarle a quien la tomó.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Hay 6 versiones: la segunda corrida registró otras tres (4, 5 y 6) con los mismos modelos. En producción sigue la versión 2. Es lo esperado: la regla pide un recall estrictamente mayor, y la versión 5 (el mismo gradient boosting) empata con la 2 (0.661), así que queda `rechazada`. Igual que la 4 y la 6. Volver a entrenar lo mismo no mueve el alias, pero llena el registro de versiones repetidas: en la práctica conviene no registrar un candidato que no cambia nada.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una versión nueva de `grantmatch` pasa a producción solo si su AUC de prueba es mayor que la de la versión en producción con la misma división (año de corte 2020) y la misma huella de datos, que quedan como parámetros de la corrida; si `pytest psets/proyecto` pasa; y si con el umbral de 0.10 no pierde recall en la validación por award. Lo decido yo, a mano, con `python -m servicio.promover <versión>`, que deja en la versión las etiquetas `promovida_por` y `reemplaza_a`: la decisión queda en el registro de MLflow, y el criterio en `docs/despliegue.md`. Por eso la versión 2 (200 árboles, AUC 0.9010) no pasó y en producción está la 1 (100 árboles, AUC 0.9018).

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Porque la versión viaja con el modelo: `python -m servicio.exportar_modelo` exporta `grantmatch@produccion` a `servicio/modelo.joblib` y escribe su número en `servicio/modelo_version.txt`, y el Dockerfile copia los dos archivos a la imagen. Al arrancar, el servicio lee ese número y lo responde en `/salud` y en cada predicción (`"version_modelo": "1"`). El contenedor nunca le pregunta a MLflow; si se quiere otra versión, se exporta de nuevo y se reconstruye la imagen.

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Que falla en silencio: sigue respondiendo rápido y con código 200, pero sus predicciones se alejan de la realidad, porque los datos de producción cambian (deriva de datos o de concepto) y el modelo quedó con lo que vio al entrenar. Ni la latencia ni `/salud` lo detectan, porque miden el servicio, no la calidad. Me enteraría comparando las predicciones con lo que pasó de verdad cuando llega la etiqueta (la tarifa cobrada, el contracargo de un fraude), con un retraso; y mientras tanto, con señales indirectas: la deriva de las features de entrada y de la distribución de las predicciones respecto al entrenamiento (con PSI o KS), y la calidad de los datos (columnas vacías o fuera de rango). Todo con alertas, no mirando a mano.
