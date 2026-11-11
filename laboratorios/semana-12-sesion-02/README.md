# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

Entrenar y evaluar el modelo en el notebook (una regresión lineal de salarios, con R² de 0.27, porque el foco no es la calidad) y **guardarlo en un archivo** con `joblib.dump` (`lin_regress.sav`). Después, **escribir una API REST con FastAPI** (`main.py`): un esquema de entrada con Pydantic, el modelo cargado una vez al arrancar, una ruta GET de bienvenida y una POST `/predict` que convierte la petición en un DataFrame, predice y devuelve el salario en JSON. **Levantarla con uvicorn y probarla**: abrir la dirección local y mandarle una petición con `requests`. Por último, **empaquetarla en un contenedor**: un `Dockerfile` (imagen base de Python, `requirements.txt`, copiar el código, exponer el puerto 8000, el comando de uvicorn), `docker build` y `docker run -p 8000:8000`.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que la API corra igual en cualquier máquina, no solo en la suya. El ejemplo es un equipo de 20 personas que necesitan usarla: cada uno tiene otro sistema operativo, otras versiones de librerías y otras herramientas, y replicar el entorno a mano es difícil. La imagen empaqueta la aplicación con sus dependencias, así que basta con tener Docker y acceso a la imagen, sin instalar nada más ni igualar versiones. Además es más rápido y liviano que un entorno o una máquina virtual.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio de predicción encendido todo el tiempo: una API (por ejemplo FastAPI) que carga el modelo una sola vez al arrancar y queda escuchando en una dirección que la app alcanza por HTTP, con la misma preparación y la misma función de features del entrenamiento. Cuando el pasajero elige el destino, la app le manda el viaje como lo conoce, en JSON: fecha y hora de inicio, zona de origen, zona de destino, distancia, pasajeros y método de pago. No manda `hora_sin`, `hora_cos`, `es_hora_pico` ni `es_tarifa_fija`: esas las calcula el servicio igual que se calcularon para entrenar. El servicio responde la tarifa estimada en la misma respuesta, en menos de un segundo, porque el pasajero está mirando la pantalla.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo está en verde y el servicio no predice nada: `/predecir` responde `{"tarifa_estimada": 0.0}` a cualquier viaje, porque la tarifa está fija en 0.0 y el modelo nunca se llama. `/salud` solo dice que el archivo del modelo se cargó, no que se use bien. Las pruebas pasan porque las de `/predecir` todavía no tienen `assert`. Además el esquema solo pide las dos zonas, así que acepta viajes sin fecha ni distancia. Un 200 y un `3 passed` no dicen que la respuesta sea correcta.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Al viaje sin `distancia_km` le responde 200 con una tarifa: es opcional en el esquema (`None` por defecto), y el imputador del pipeline la completa con la mediana del entrenamiento (con el viaje Midtown → JFK da 68.56 en vez de 70.39). Al viaje sin `zona_origen` le responde **422**: es obligatoria, y FastAPI devuelve `"Field required"` con `loc` = `["body", "zona_origen"]`.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo daría tarifas equivocadas sin ningún error. Recibiría un `es_hora_pico` que no significa lo que aprendió: a las 8:30 de un día de semana podría llegarle 0 y estimaría como en hora valle. Todo responde 200 y nadie se entera (es un desfase entre entrenamiento y servicio). Por eso el servicio no recalcula las features por su cuenta: importa `agregar_features` de `features_taxis.py`, la misma función que construyó `viajes_features.csv`.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce FastAPI, no mi código. Antes de llamar a `predecir`, FastAPI valida el JSON contra el esquema `Viaje` con Pydantic; como falta un campo obligatorio, responde 422 con el detalle del error y la función nunca se ejecuta. La petición no llega al modelo: el contrato de entrada se cumple antes de gastar una predicción.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque uvicorn, sin `--host`, escucha en `127.0.0.1`, y dentro del contenedor esa dirección es el localhost del propio contenedor (el log lo dice: `Uvicorn running on http://127.0.0.1:8000`). `docker run -p 8000:8000` reenvía las conexiones a la interfaz de red del contenedor, no a su localhost, y ahí nadie escucha, así que `curl` falla aunque el servicio esté vivo. Con `--host 0.0.0.0` escucha en todas las interfaces y el puerto publicado llega.

**Lo que respondió el servicio:**

Viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta) con el modelo de `entrenar.py`: `{"tarifa_estimada": 70.39}`, con código 200. Viaje sin zona de origen: código 422, `{"detail": [{"type": "missing", "loc": ["body", "zona_origen"], "msg": "Field required", "input": {"fecha_hora_inicio": "2026-11-05 08:30", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}}]}`. `pytest -v`: 3 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

Pago de madrugada (02:40, app, tarjeta nueva, 2 intentos fallidos, 0 viajes previos): `{"probabilidad_fraude": 0.9996, "revisar": true}`. Pago de la tarde (14:10, tarjeta conocida, 22 viajes previos): `{"probabilidad_fraude": 0.0159, "revisar": false}`. `python comprobar.py`: pasos 1 a 5 OK.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no necesita saber el umbral ni aplicar la regla: solo hace lo que dice `revisar`. La decisión de negocio vive en un solo lugar, igual para todas las aplicaciones que llamen, y la app no tiene que cambiar cuando cambia la regla. Si mañana el umbral es 0.20, se cambia `UMBRAL` en `servicio_fraude.py` (mejor si se lee de una variable de entorno, sin tocar código) y se vuelve a desplegar el servicio (reconstruir la imagen y reiniciar el contenedor). La app y el modelo no cambian, y no hay que reentrenar. La prueba del pago válido sigue pasando, porque 0.25 también alcanza 0.20.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí. Lo comprobé con el pago 70058 de `pagos_ayer.csv`: el lote le dio 0.9984 y el servicio, con los mismos datos, 0.9984. Para que coincidan tiene que ser igual el modelo (el mismo archivo `modelo_fraude.joblib`, la misma versión), la preparación (las mismas `agregar_features` y `columnas` de `preparacion_pagos.py`, importadas y no copiadas), los datos de entrada del pago y las versiones de las librerías (scikit-learn 1.5.2 en los dos).

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde **422** con `{"detail": [{"type": "float_parsing", "loc": ["body", "monto"], "msg": "Input should be a valid number, unable to parse string as a number", "input": "cuarenta"}]}`. Lo produce la línea `monto: float` del esquema `Pago`: Pydantic intenta convertir `"cuarenta"` en número, no puede, y FastAPI rechaza la petición antes de llamar a `evaluar_pago`.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe el par como lo conoce quien llama: de la convocatoria, `award_field_name`, `funder_country_code`, `amount_usd`, `duration_days`, `funding_type`, `award_domain_name` e `is_global_south`; del investigador, `work_field_name`, `author_country_codes` (una lista de países), `work_domain_name`, `authors_count` y `cited_by_count`. El servicio calcula `mismo_campo` y `funder_en_pais_autor`, y convierte `is_global_south` a 0/1. Sé que las calcula igual que en el entrenamiento porque `servicio/api.py` no las reescribe: importa `agregar_features` de `modelado/pipeline_proyecto.py`, la misma función que usan los notebooks de entrenamiento, y le pasa los países con el mismo formato `"US|DE"` de la OBT. Además, `test_el_servicio_calcula_las_features_como_el_entrenamiento` en `servicio/test_api.py` comprueba los dos valores a mano y corre en GitHub Actions en cada push.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Cada respuesta del servicio lleva `version_modelo`, el `run_id` de MLflow del modelo exportado (lo escribe `exportar_modelo.py` en `modelo_info.json`). Si además guardo cada predicción con su `run_id`, la fecha y la entrada, ante un reclamo busco esa fila y el `run_id` me lleva en MLflow a la corrida exacta: hiperparámetros, huella de los datos, año de corte, métricas y el pipeline guardado. Para volver al modelo anterior no hace falta reentrenar: MLflow guarda el artefacto de cada corrida, así que corro `python servicio/exportar_modelo.py <run_id anterior>`, reconstruyo la imagen y reinicio el contenedor (y muevo la etiqueta `candidato` = `final`). Más seguro todavía es etiquetar cada imagen con su `run_id` (`grantmatch-servicio:<run_id>`), para volver con un solo `docker run`, o usar el registro de modelos de MLflow, con versiones numeradas y un alias que apunte a la que está en producción.
