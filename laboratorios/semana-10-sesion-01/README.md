# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un grupo de modelos (estimadores o miembros) cuyas predicciones se combinan en una sola, como un comité: si los miembros son diversos, el conjunto suele acertar más que cualquiera por separado. En **bagging** los modelos se entrenan de forma independiente, cada uno con una muestra bootstrap de los datos (con reemplazo), y se combinan por promedio o votación, con el mismo peso para todos; el ejemplo es el bosque aleatorio, que además muestrea columnas. En **boosting** se entrenan en serie: cada modelo nuevo le da más peso a los ejemplos que los anteriores fallaron, y cada uno pesa en el resultado según qué tan bien le fue; los ejemplos son AdaBoost y gradient boosting, que ajusta cada árbol nuevo al error residual.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque cada árbol se equivoca de forma distinta: se entrena con una muestra distinta de filas (y de columnas en cada corte), así que sus errores están poco correlacionados. Al promediar, los errores que van en direcciones opuestas se cancelan y la varianza baja, mientras el sesgo queda más o menos igual. El mejor árbol solo también tiene su propio ruido; el promedio se queda con lo que todos los árboles tienen en común, que es la señal.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: aprendió también el ruido y los casos particulares del entrenamiento, así que memoriza en vez de generalizar. Lo delata la brecha entre los dos errores. Se corrige con un modelo más simple o regularizado (menos profundidad, menos columnas, Ridge), con más datos, o promediando muchos modelos.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperaba un error grande, del orden de la dispersión de la tarifa: la línea base se equivoca 12.53 en promedio en prueba. Sirve como piso de comparación: cualquier modelo que use columnas tiene que equivocarse menos que esto, o no está aprendiendo nada útil.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

Error casi 0 en entrenamiento, porque memoriza cada viaje (0.0029), y bastante peor en prueba (1.70). Es sobreajuste: aprende también el ruido.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, la brecha baja de 1.70 a 0.26 y el error en prueba mejora a 1.27: pierde un poco en entrenamiento y gana en generalización. El bosque de 200 árboles baja el error en prueba respecto al árbol sin límite (1.30 contra 1.70), porque promediar reduce la varianza, pero su brecha sigue siendo 0.83: cada árbol del bosque crece sin límite. En estos datos no supera al árbol de profundidad 8.

**Lo que confirmó la tabla, TODO por TODO:**

Línea base: entrenamiento 13.20, prueba 12.53, brecha -0.67. Árbol sin límite: 0.0029, 1.70, brecha 1.70. Árbol con max_depth=8: 1.01, 1.27, brecha 0.26. Bosque de 200 árboles: 0.47, 1.30, brecha 0.83.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

3275 columnas contra 2380 filas de entrenamiento: hay más coeficientes que datos. La regresión puede ajustar el entrenamiento casi exacto (error 0.91), pero los coeficientes quedan enormes e inestables, y en prueba el error explota (del orden de 2.4 × 10⁹).

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento Ridge empeora un poco (0.91 a 1.03); en prueba pasa de unos 2.4 × 10⁹ a 1.32. Ridge penaliza los coeficientes grandes, así que ya no puede ajustar cada detalle del entrenamiento, que era en buena parte ruido. Eso cuesta un poco en entrenamiento y es justo lo que lo hace generalizar.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

Gradient boosting, con 1.28 de error en prueba. La línea base se equivoca 12.53: el mejor modelo se equivoca unos 11.26 dólares menos por viaje, alrededor de un 90 % menos.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La línea base predice siempre la mediana del entrenamiento y no aprende nada, así que su error depende solo de qué tan lejos de esa mediana quedan las tarifas de cada conjunto. En esta división, por azar, la prueba quedó con menos viajes de tarifa fija de 70 (8.9 % contra 10.4 % en entrenamiento) y con tarifas menos dispersas (desviación 17.3 contra 18.2), así que queda más cerca de la mediana y su error sale menor.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py` (`psets/proyecto/modelado/`), que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

`DummyClassifier(strategy="prior")`: predice siempre la proporción de pares reales del entrenamiento. Es un problema de clasificación con clases balanceadas, así que su AUC es 0.5 y su exactitud 0.5: es lo que saca cualquier modelo que no distinga un financiamiento real de un par al azar. No hay una regla del dominio más natural; la más cercana sería "recomendar solo awards del mismo campo", y esa ya está como feature (`mismo_campo`).

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

⟨Revisar con la tabla real.⟩ El candidato es el bosque sin límite: AUC de entrenamiento ⟨X⟩ contra ⟨X⟩ en prueba, una brecha de ⟨X⟩, porque cada árbol crece hasta memorizar los pares. La regresión logística casi no tiene brecha (⟨X⟩): con pocas columnas no tiene con qué memorizar.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Dos. En la regresión logística probé `C=1` y `C=0.01` (más regularización): ⟨la AUC de prueba casi no cambia, ⟨X⟩ contra ⟨X⟩, porque el modelo no se sobreajustaba⟩. En el bosque limité la profundidad a 8: la brecha bajó de ⟨X⟩ a ⟨X⟩ y la AUC de prueba ⟨subió/se mantuvo⟩ en ⟨X⟩.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor en prueba es ⟨modelo⟩, con AUC ⟨X⟩ contra 0.5 de la línea base (⟨+X⟩). En exactitud, ⟨X⟩ contra 0.5. Ojo: los negativos son pares al azar, que son fáciles de distinguir; con negativos más difíciles (awards del mismo campo que no financiaron la publicación) la mejora sería menor.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error: el costo por predicción y por mes, el tiempo de respuesta (la app tiene que dar la tarifa al instante), si la misma consulta da siempre el mismo número (no se le puede cobrar distinto a dos pasajeros por el mismo viaje), si se puede explicar una tarifa, el costo y el tiempo de entrenar y reentrenar, y qué datos salen de la empresa (un LLM externo recibe los datos de cada viaje).
