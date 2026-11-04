# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

_(tu respuesta)_

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

El costo. GrantMatch tiene que evaluar muchísimas combinaciones investigador–award (la OBT tiene 1,88 millones de pares reales solo con una parte de OpenAlex), y un LLM cobra por cada consulta: un modelo clásico, una vez entrenado, evalúa millones de pares casi gratis. La explicabilidad pesa en segundo lugar, porque al recomendar un award hay que poder decir por qué (mismo campo, mismo país del funder).

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error: el costo por predicción y por mes, el tiempo de respuesta (la app tiene que dar la tarifa al instante), si la misma consulta da siempre el mismo número (no se le puede cobrar distinto a dos pasajeros por el mismo viaje), si se puede explicar una tarifa, el costo y el tiempo de entrenar y reentrenar, y qué datos salen de la empresa (un LLM externo recibe los datos de cada viaje).

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Que el optimizador llegó al tope de iteraciones (200 por defecto) sin converger: los pesos todavía estaban cambiando cuando lo detuvo. La red quedó a medio entrenar, por eso salía peor (1.69 de error en prueba). Con 1000 iteraciones esperaba que convergiera y bajara el error, y así fue: 1.17 en prueba, mejor incluso que el gradient boosting (1.28).

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

1729 números. De la entrada a la capa oculta: 25 × 64 = 1600 pesos más 64 sesgos; de la capa oculta a la salida: 64 pesos más 1 sesgo. 1600 + 64 + 64 + 1 = 1729.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La distancia, y así fue: 0.98 de importancia. Le siguen `es_tarifa_fija` (0.012) y `es_hora_pico` (0.003). Para explicarle al conductor una tarifa de 70, el ensamblaje permite responder: sus importancias y sus árboles muestran que el viaje cae en `es_tarifa_fija` = 1. En la red, los 1729 pesos no tienen un significado que se pueda leer.

**Lo que confirmó la salida, TODO por TODO:**

Gradient boosting: error 1.07 en entrenamiento y 1.28 en prueba, 0.26 s. Red neuronal con 1000 iteraciones: 1.09 y 1.17, 2.92 s (11 veces más lenta). Parámetros de la red: 1729. Columnas más importantes del ensamblaje: `distancia_km` (0.9824), `es_tarifa_fija` (0.0120), `es_hora_pico` (0.0028).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.90 | 0.27 s de cómputo | ~0 (corre en el servidor) | En parte: importancia de columnas |
| Red neuronal | 0.96 | 3.35 s de cómputo | ~0 | No: 1729 pesos sin significado legible |
| LLM (Claude Opus 5.5) | 1.16 | Nada que entrenar | ~1.45 dólares (726 tokens por prompt de 20 viajes, 500 prompts, 4 dólares por millón de tokens de entrada) | Da una explicación en texto, sin garantía de que sea la razón real |

Usé Claude Opus 5.5.

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

⟨Completar con la segunda corrida.⟩ En una conversación nueva el LLM ⟨no dio⟩ los mismos 20 números: ⟨X⟩ de 20 cambiaron, hasta ⟨X⟩ dólares en el mismo viaje. Para una app que cobra es un problema serio: dos pasajeros que piden el mismo viaje recibirían precios distintos, y no se puede auditar ni reproducir una tarifa cobrada.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor fue un viaje de JFK Airport a JFK Airport de 1.01 km: real 11.46, el LLM dijo 8.20 (error de 3.26). Es un viaje corto dentro del aeropuerto, y ninguno de los 30 ejemplos del prompt se le parecía: el LLM aplicó la regla general de la distancia. El ensamblaje, con las 2380 filas de entrenamiento, sí vio viajes de ese tipo y se acercó más: predijo 12.74.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque cada consulta al LLM cuesta dinero y tiempo, y en un chat hay que copiar y pegar a mano: 595 viajes serían unos 30 prompts. Una comparación sobre 20 viajes es poco confiable: un solo viaje raro (como el de JFK a JFK) mueve el promedio unos 0.16. La diferencia entre el ensamblaje (0.90) y la red (0.96) no es concluyente con 20 viajes; la del LLM (1.16) apenas lo es.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

El código está en [`proyecto/comparar_proyecto.py`](./proyecto/comparar_proyecto.py) y escribe `proyecto/comparacion_proyecto.csv`. La prueba del LLM usa 10 pares de prueba (5 reales y 5 al azar) y 20 de entrenamiento como ejemplos; la métrica sobre esas 10 filas es la exactitud, porque una AUC con 10 filas no dice nada.

| Opción | AUC en prueba | Exactitud en los 10 pares | Entrenamiento | Costo de 10 000 pares | Explicabilidad |
|---|---|---|---|---|---|
| ⟨mejor ensamblaje⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ s | ~0 | Importancia de columnas |
| Red neuronal (64 neuronas, ⟨X⟩ parámetros) | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ s | ~0 | No |
| LLM (⟨modelo⟩) | — | ⟨X⟩ | — | ⟨X⟩ dólares | Texto, sin garantía |

Usaría el ensamblaje. ⟨Ajustar con los números reales:⟩ su AUC en prueba (⟨X⟩) es igual o mejor que la de la red (⟨X⟩), entrena ⟨X⟩ veces más rápido y sus importancias muestran qué pesa en cada recomendación: `mismo_campo` y `funder_en_pais_autor` arriba, que es justo lo que un investigador querría saber de por qué se le recomienda un award. El LLM no conviene como clasificador: acertó ⟨X⟩ de 10, cuesta ⟨X⟩ dólares por cada 10 000 pares (y GrantMatch tiene que evaluar millones de combinaciones investigador–award), no da el mismo resultado dos veces y manda los datos fuera. Donde sí tendría sentido un LLM es en el texto (resúmenes y descripciones de awards), que el ensamblaje no usa.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

La versión exacta de los datos (qué archivo y qué limpieza), la versión del código (el commit), las versiones de las librerías (Python, scikit-learn, numpy), las semillas (`random_state` de la división y del modelo) y cómo dividieron los datos. Lo vi en este mismo curso: el mismo `ejercicio2_regularizacion.py` daba un error de prueba de 22.7 para `lineal grado 3` con scikit-learn 1.9 y de 2.4 × 10⁹ con la 1.5.2 del curso.
