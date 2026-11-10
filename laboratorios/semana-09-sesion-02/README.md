# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano tu notebook de features de la sesión pasada. Los tres ejercicios de hoy son scripts, no notebooks: un pipeline de preparación es código de producción, que tiene que correr igual cada vez y que otros archivos importan. En el Ejercicio 3, la preparación de tu proyecto pasa del notebook donde la probaste a ese script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

Es cuando información del conjunto de prueba (o del objetivo) entra por error en el entrenamiento. El modelo aprende patrones que en la realidad no tendrá disponibles, así que sus métricas de evaluación salen infladas; en producción esa información no está, el modelo no generaliza y su rendimiento cae. El artículo lo compara con un estudiante que practica con el examen filtrado: saca muy buena nota en ese papel, pero no refleja lo que sabe.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La fuga de la variable objetivo (*target leakage*): usar como feature algo que es consecuencia del objetivo y que no existe en el momento de predecir. El ejemplo del artículo son las llamadas al servicio al cliente en un modelo de cancelación de suscripciones, que ocurren por la cancelación. La duración del viaje es lo mismo: se conoce recién cuando el viaje termina, igual que la tarifa que queremos predecir.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si uso todos, la media y la desviación ya contienen información de los viajes de prueba, y la evaluación deja de simular viajes nuevos: es fuga de información. En los números cambia poco (la media de distancia pasó de 8.5034 con todos a 8.5576 solo con entrenamiento en la Semana 9.2), pero el procedimiento queda mal, y con datos más chicos o con transformaciones más fuertes la diferencia puede ser grande.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En `X_preparado = escalador.fit_transform(imputador.fit_transform(X))`: el imputador y el escalador aprenden con las 2975 filas antes de dividir, así que la mediana, la media y la desviación ya incluyen el conjunto de prueba. Lo delata la salida: `Filas que vio el escalador: 2975`, cuando el entrenamiento debería tener 2380.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Mide el error de entrenamiento, no el de prueba: el modelo se evalúa con las mismas filas con las que aprendió. Ese número (2.5590) no dice nada de cómo le irá con viajes nuevos.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba un cambio pequeño, y así fue: 2.5375 con fuga contra 2.5373 corregido. Que casi no cambie no significa que no había fuga: la prueba igual entró al entrenamiento (2975 filas contra 2380). Aquí el efecto es mínimo porque la media y la mediana casi no cambian con 595 filas más (8.5034 contra 8.5576), pero con otro dataset o con otra transformación la diferencia puede ser grande.

**Lo que confirmó la salida, TODO por TODO:**

Versión del compañero: el escalador vio 2975 filas, media de distancia 8.5034, error en prueba 2.5375. Versión correcta: el escalador vio 2380 filas (solo entrenamiento), media 8.5576, 595 filas de prueba, error en prueba 2.5373.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

25 columnas. Las 2 numéricas siguen siendo 2 (imputadas y escaladas); las 5 ya listas pasan sin cambios; y las 3 categóricas se convierten en 18 columnas one-hot: 8 de `zona_origen`, 8 de `zona_destino` y 2 de `metodo_pago`. 2 + 5 + 18 = 25.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento: distancia 6.94 y pasajeros 1.0, mientras que la mediana de distancia de todo el dataset es 6.93. Demuestra que el imputador aprendió solo con entrenamiento: el conjunto de prueba no influyó en la preparación.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Como el entrenamiento nunca vio el Bronx, `OneHotEncoder(handle_unknown="ignore")` deja las 8 columnas de `zona_origen` en 0, y el viaje se predice con el resto de las columnas. Con `pd.get_dummies` aplicado solo a ese viaje se crearía una columna `zona_origen_Bronx` que el modelo no conoce y faltarían las otras 8, así que las columnas no coincidirían con las del entrenamiento y el modelo fallaría o leería las columnas desalineadas.

## Ejercicio 3: el pipeline de tu proyecto, como script

Copia la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py) a `psets/proyecto/modelado/`, al lado de tu notebook de features. Desde la próxima sesión, tus notebooks de modelos importan este archivo en vez de copiar su código:

```python
from pipeline_proyecto import cargar_y_dividir, construir_preparacion
```

### Paso 1: pasa tus features a `agregar_features` (5 minutos)

Copia de tu notebook de la sesión pasada solo las líneas que construyen tus 2 features, dentro de la función `agregar_features`. Sin gráficos, sin `print` y sin las celdas que usaste para probar ideas: esas se quedan en el notebook, como evidencia de por qué elegiste cada feature.

### Paso 2: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

- **Objetivo:** `label` (1 si el award financió la publicación, 0 si es un par armado al azar).
- **Numéricas:** `amount_usd`, `duration_days`, `authors_count`, `cited_by_count`.
- **Categóricas:** `funding_type`, `award_domain_name`, `work_domain_name`.
- **Ya listas (0/1):** `mismo_campo`, `funder_en_pais_autor`, `is_global_south`.
- **Fuera, porque no existen al predecir o filtrarían la respuesta:** `funded_outputs_count` (se sabe después del financiamiento), `publication_year` (en un par real la publicación sale después del award, así que delataría el `label`), `fwci`, los identificadores (`award_key`, `work_key`) y los nombres (`funder_name`, campos en texto ya resumidos en `mismo_campo`).

### Paso 3: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha dentro de `cargar_y_dividir`; si no, al azar. Justifica.

Sí: el modelo se va a usar con convocatorias nuevas, así que divido por fecha. Entrenamiento con los awards que empezaron antes del año de corte (el percentil 80 de `award_start_year`) y prueba con los de ese año en adelante. Así la prueba simula convocatorias que el modelo nunca vio. Además, cada award y su par negativo comparten el año, así que caen siempre del mismo lado: un mismo award no aparece en entrenamiento y en prueba a la vez.

### Paso 4: corre el script (10 minutos)

Desde una terminal, en `psets/proyecto/modelado/`:

```
python3 pipeline_proyecto.py
```

Si corre de principio a fin en una terminal nueva, sin el notebook abierto, otro código ya lo puede importar. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

Desde `psets/proyecto/modelado/`: año de corte: ⟨X⟩. Entrenamiento: ⟨filas, columnas⟩. Prueba: ⟨filas, columnas⟩. Proporción de pares reales: 0.5 en los dos conjuntos.

Columnas que produce: las 4 numéricas escaladas (`numericas__amount_usd`, `numericas__duration_days`, `numericas__authors_count`, `numericas__cited_by_count`), una columna one-hot por cada valor de `funding_type`, `award_domain_name` y `work_domain_name` (⟨X⟩ en total), y las 3 ya listas (`ya_listas__mismo_campo`, `ya_listas__funder_en_pais_autor`, `ya_listas__is_global_south`).

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: aprendió también el ruido y los casos particulares del entrenamiento, así que memoriza en vez de generalizar. Lo delata la brecha entre los dos errores. Se corrige con un modelo más simple o regularizado (menos profundidad, menos columnas, Ridge), con más datos, o promediando muchos modelos.
