# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

El código, los datos y el modelo. El código con Git, los datos con DVC (Data Version Control), que está hecho para datasets grandes y deja en Git solo unos archivos `.dvc` con los metadatos, y el modelo con MLflow.

**¿Por qué no alcanza con Git para versionar los datos?**

Porque Git está hecho para archivos de texto chicos que cambian línea por línea. Los datos suelen pesar cientos de megas o gigas, cambian enteros (un nuevo snapshot), y Git guarda una copia completa de cada versión en el historial de todos los que clonan el repositorio. Además, muchos datos no se pueden subir (privados, con licencia, como los de este curso, que están en `.gitignore`). Por eso en el repositorio va el código y una referencia a los datos (de dónde salen y su huella), y los datos viven en otro lado.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

La versión exacta de los datos (qué archivo; su huella), cómo dividieron entrenamiento y prueba (la semilla o el criterio), los hiperparámetros y la semilla del modelo, la versión del código (el commit) y las versiones de las librerías. En el Ejercicio 1 la diferencia era la semilla de la división: 42 contra 7, con los mismos datos y el mismo modelo.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Los dos "tienen razón": cada número es el error real de ese modelo sobre su propio conjunto de prueba (1.2754 con una división, 1.1867 con la otra). La salida dice el error y el nombre del modelo, y que las divisiones fueron 42 y 7, pero solo porque `evaluar` lo imprime; no queda guardado en ningún lado y no dice si los datos eran los mismos ni con qué hiperparámetros se entrenó cada uno. Sin eso no se puede saber por qué difieren.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La división: con otra semilla, los 595 viajes de prueba son otros, y unos son más fáciles de estimar que otros. Esperaba ver distinto `semilla_division` (42 contra 7) y todo lo demás igual. Así fue: la huella de los datos es la misma en las tres corridas (`4800204d50c0`) y `n_estimators` es 100 en los dos gradient boosting.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder usar exactamente el modelo que produjo ese número sin volver a entrenarlo: servirlo en producción, revisarlo o compararlo después. Volver a entrenar no garantiza el mismo modelo (otra versión de scikit-learn, otro orden de los datos, otra semilla). Con el artefacto guardado, el número registrado y el modelo que se usa son el mismo objeto.

**Lo que mostró la tabla del paso 5:**

| Corrida | Error en prueba | Semilla de la división | Huella de los datos |
|---|---|---|---|
| gradient boosting | 1.1867 | 7 | `4800204d50c0` |
| gradient boosting | 1.2754 | 42 | `4800204d50c0` |
| bosque aleatorio | 1.3016 | 42 | `4800204d50c0` |

Los dos gradient boosting tienen los mismos datos y los mismos hiperparámetros: la diferencia de 0.09 sale solo de la división.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La red neuronal, con 1.1695 de error en prueba. Le gana al gradient boosting (1.2754) por 0.106, un 8 % menos. Detrás quedan el bosque (1.3016), Ridge (2.5565) y la línea base (12.5308).

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que el modelo guardado es el mismo que produjo el número registrado: con los mismos datos de prueba da exactamente el mismo error (1.169505 en los dos casos). No garantiza que sea el mejor modelo de verdad: la diferencia con el gradient boosting sale de una sola división (y vimos que la división sola mueve el error 0.09), ni que siga funcionando igual con viajes nuevos o con datos que cambien.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por la huella de los datos: la nueva corrida tendría otro `huella_datos` (el MD5 del CSV cambia con un solo byte), aunque los hiperparámetros y la semilla sean los mismos. Comparando ese parámetro entre corridas se ve que los números no salen de los mismos datos.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Todos sus hiperparámetros, incluidos los que no escribí (el optimizador `adam`, la tasa de aprendizaje, `alpha`, `max_iter`), que tardó 2.6 segundos contra 0.2 del gradient boosting, con qué datos y qué división se entrenó, y el modelo mismo, con las versiones de las librerías con las que se guardó. Una tabla escrita a mano tendría solo el nombre y el error.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

⟨Completar al correr [`experimentos_proyecto.ipynb`](../../psets/proyecto/modelado/experimentos_proyecto.ipynb) con los datos reales.⟩ Elegí ⟨modelo⟩, corrida `⟨run_id⟩`, con AUC de prueba ⟨X⟩ contra ⟨X⟩ del segundo (⟨modelo⟩) y 0.5 de la línea base. La AUC de entrenamiento es ⟨X⟩ (brecha ⟨X⟩), y entrena en ⟨X⟩ s. Al cargarlo desde el registro reproduce exactamente la AUC registrada.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita: (1) los datos, que no están en el repositorio: correr `exportar_obt.sql` sobre la misma OBT y guardar el resultado en `psets/proyecto/data/raw/`; (2) el código en el mismo commit (los notebooks, `pipeline_proyecto.py` y la semilla 42 con la que se arman los pares negativos); (3) las mismas librerías, que da el Dev Container del curso; y (4) correr en orden `features_proyecto.ipynb` y `experimentos_proyecto.ipynb`.

Queda registrado en MLflow: los hiperparámetros, el año de corte de la división, la huella de `pares_proyecto.csv`, las métricas, el modelo entrenado y las versiones de las librerías con las que se guardó. No queda: los datos mismos (solo su huella), la consulta que los generó, la semilla de los negativos ni el commit del código. Además, la muestra de Snowflake (`sample (50000 rows)`) es aleatoria: una nueva exportación da otras filas y otra huella. Para el mismo número exacto hay que guardar el CSV exportado o fijar la semilla del `sample`.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Bastante para lo que separa a los modelos. El mismo gradient boosting dio 1.2754 con la división de semilla 42 y 1.1867 con la de semilla 7: casi 0.09 de diferencia solo por cambiar qué viajes caen en prueba, más que lo que separaba al gradient boosting del bosque (1.2754 contra 1.3016). En el registro se ve en la tabla del Ejercicio 1 de la Semana 11.1: las dos corridas de gradient boosting tienen la misma huella de datos y los mismos hiperparámetros, y solo cambia `semilla_division`. Con validación cruzada al azar, los cinco pliegues del gradient boosting van de 1.16 a 1.28.
