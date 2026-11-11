# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

Lo que los distingue es **cuándo se calcula la predicción**. En batch se calcula antes de que llegue la petición: un proceso corre cada cierto tiempo (cada pocas horas o una vez al día), predice para todos y guarda los resultados, y cuando alguien pregunta solo se lee la predicción ya hecha. En online se calcula después de que llega la petición, en ese momento. El artículo separa eso de cómo se *consume* la predicción: una predicción batch se puede servir en tiempo real por una API que solo busca el resultado guardado de ese usuario, y sigue siendo batch. Ejemplo batch: las recomendaciones de restaurantes de DoorDash, de subreddits de Reddit o las de Netflix (2021), calculadas de antemano con filtrado colaborativo. Ejemplo online: una recomendación que usa lo que el usuario hizo en la sesión, como cuántas veces vio algo en los últimos 30 minutos, calculada cuando llega la petición.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

La predicción batch llega con retraso: se calculó en el último lote y no refleja lo que pasó después (lo que el usuario acaba de hacer, un caso nuevo). Además calcula predicciones redundantes que nadie usa, para usuarios que no vuelven, y eso cuesta cómputo. El online las resuelve porque calcula solo lo que se pide y con los datos de ese momento. Lo que cuesta: la latencia de inferencia se vuelve crítica (hay alguien esperando la respuesta), las features en tiempo real necesitan infraestructura de streaming, que el artículo dice que es difícil de montar, hacen falta buenos embeddings, y en general más infraestructura que un lote, que es más simple y barato.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa de un viaje que se pide en el momento no se puede calcular de noche para todos los casos posibles: la distancia y la hora son continuas y las combinaciones de zonas, horas, distancia, pasajeros y método de pago son demasiadas. Solo los viajes reservados para el día siguiente se conocen de antemano, y esos sí se pueden calcular de noche (es el Ejercicio 1). El fraude tampoco se puede precalcular: depende de datos que existen recién en el momento del cobro (tarjeta nueva, intentos fallidos, el monto). El que tiene que responder en el momento es el de fraude, y dentro del cobro: del orden de unos cientos de milisegundos, porque el pago está esperando la decisión. La tarifa de un viaje no reservado también se pide en el momento, pero con menos urgencia: menos de un segundo mientras el pasajero mira la pantalla.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque una prueba de pytest pasa si no lanza ninguna excepción, y las funciones de los `TODO 4` y `TODO 5` todavía no tienen ningún `assert`: llaman a `predecir_lote` y no comprueban nada, así que pasan aunque la tabla no tenga tarifas. La única que falla es `test_columnas_de_salida`, la única con un `assert`, porque la tabla tiene solo `id_viaje`. "Passed" quiere decir "no se rompió", no "está bien": una prueba sin `assert` no prueba nada.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Que les estime una tarifa a todos. El modelo guardado es el pipeline completo, con la preparación adentro: `SimpleImputer` completa `pasajeros` con la mediana y `metodo_pago` con la categoría más frecuente, aprendidas en el entrenamiento, así que el modelo nunca recibe un vacío. Saltarlos sería peor: esos viajes están reservados y por la mañana la app no tendría tarifa que mostrarles. Así pasó: los 4 viajes sin pasajeros y los 2 sin método de pago salieron con tarifa y ninguna fila quedó vacía.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para saber qué modelo produjo cada estimación. La tabla se lee al día siguiente y las estimaciones se guardan; cuando el modelo se reentrena, filas de noches distintas vienen de modelos distintos. Si un pasajero reclama por una tarifa, si se comparan dos versiones o si hay que volver al modelo anterior, la versión en la fila (la huella MD5 del archivo `.joblib`) dice exactamente qué archivo la calculó, sin depender de recordar cuándo se cambió.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

La del `TODO 4` falla si `predecir_lote` pierde, agrega o reordena viajes: por ejemplo un `viajes.dropna()` antes de predecir (se saltaría el viaje 12, sin pasajeros) o un `sort_values("tarifa_estimada")` en la salida. La del `TODO 5` falla si algún viaje queda sin tarifa (predecir sin la preparación que imputa, o reemplazar los vacíos por NaN en vez de estimarlos) o si la columna `version_modelo` no lleva la versión que se pasó (escrita fija o vacía).

**Lo que mostró el lote:**

Versión del modelo `def9bdd14164`, 60 filas para 60 viajes. Tarifa más baja 4.67 (viaje 90014), más alta 70.26 (viaje 90051), promedio 22.87. Los viajes sin pasajeros recibieron tarifa como cualquier otro: 90023 → 33.94, 90024 → 29.88, 90025 → 25.80, 90048 → 17.58. Los dos sin método de pago también: 90044 → 60.79, 90050 → 29.59. `pytest -v`: 3 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

Con el umbral de 0.10 quedaron **26 pagos** en la cola (modelo `f3038612d4f2`; el primero es el 70058, con 0.9984). En la Semana 11 la proporción de fraude era 295 de 6000 pagos (4.9 %), así que en 300 pagos esperaría unos **15 fraudes**. La cola tiene más pagos que fraudes esperados, a propósito: con un umbral bajo se prefiere revisar algunos pagos legítimos a dejar pasar un fraude.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de frenar el fraude antes de que ocurra: cuando el lote marca el pago, el viaje ya se hizo, el cobro ya pasó (o falló) y la tarjeta robada pudo usarse otras veces durante ese día. Con el lote solo se puede investigar y reclamar después. Para que eso importe, la revisión tendría que poder actuar en el momento: bloquear el cobro, pedir otra verificación o no asignar el viaje mientras el pago está pendiente. Si la revisión es un equipo que de todos modos mira la cola al día siguiente, evaluar cada pago al instante no cambia nada.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Una prueba que no depende del modelo: con probabilidades escritas a mano se sabe de antemano la cola correcta (los pagos 4, 2 y 3, en ese orden), así que prueba solo la lógica de la cola: que el umbral incluye el límite (el 0.10 entra), que el 0.05 queda fuera y que se ordena de mayor a menor. Si se reentrena el modelo o cambian sus probabilidades, la prueba sigue valiendo, no necesita el `.joblib` (que no existe en GitHub Actions) y cuando falla el error está en `a_revisar`, no en el modelo.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con `UMBRAL = 0.50` quedan **13 pagos** en la cola (la mitad). Las cuatro pruebas siguen pasando, porque ninguna usa la constante `UMBRAL`: `test_cola_de_revision` le pasa su propio umbral (0.10) a `a_revisar`, y las demás revisan la forma de la tabla y que las probabilidades estén entre 0 y 1. Las pruebas comprueban que la cola se arma bien con cualquier umbral; qué umbral usar es una decisión de negocio que no prueban. Volví a dejar 0.10.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

Por lotes, una vez por noche, después de que el pipeline de datos actualiza la OBT en Snowflake. El número que lo decide es **24 horas**: una convocatoria está abierta semanas y preparar una postulación toma días, así que una recomendación que llega al día siguiente no le hace perder nada al investigador. Además, los datos de las features solo cambian una vez al día, así que predecir en línea no daría información más fresca. El detalle está en `psets/proyecto/docs/despliegue.md`.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio de predicción encendido todo el tiempo: una API (por ejemplo FastAPI) que carga el modelo una sola vez al arrancar y queda escuchando en una dirección que la app alcanza por HTTP, con la misma preparación y la misma función de features del entrenamiento. Cuando el pasajero elige el destino, la app le manda el viaje como lo conoce, en JSON: fecha y hora de inicio, zona de origen, zona de destino, distancia, pasajeros y método de pago. No manda `hora_sin`, `hora_cos`, `es_hora_pico` ni `es_tarifa_fija`: esas las calcula el servicio igual que se calcularon para entrenar. El servicio responde la tarifa estimada en la misma respuesta, en menos de un segundo, porque el pasajero está mirando la pantalla.
