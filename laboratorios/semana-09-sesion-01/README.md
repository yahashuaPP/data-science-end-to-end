# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tu notebook de exploración de la Semana 8, Sesión 2: las features de hoy salen de tus 3 hallazgos.

Los laboratorios de hoy siguen siendo notebooks: diseñar una feature es probar varias ideas y mirar cuál aporta. La próxima sesión, las features que se queden pasan a un script, que es lo que corre en producción.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

_(tu respuesta)_

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

_(tu respuesta)_

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Creando columnas nuevas a partir de las que ya existen. Para la tarifa fija, una columna `es_tarifa_fija` que vale 1 cuando el viaje va entre JFK y una zona de Manhattan. Para la hora, sacar la hora de `fecha_hora_inicio`, una columna `es_hora_pico` y la hora como ciclo (seno y coseno), para que 23 y 0 queden cerca.

## Ejercicio 1: features de fecha y hora

Notebook: [`starter/ejercicio1_features_tiempo.ipynb`](./starter/ejercicio1_features_tiempo.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. Ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número, 23 y 0 quedan a 23 unidades, la mayor distancia posible, aunque en el reloj están a una hora. Con seno y coseno las dos parejas (23 y 0, 11 y 12) quedan a la misma distancia, 0.26, que es lo que pasa en el reloj.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta. La tarifa depende de la distancia y también del tiempo, y en hora pico el tráfico alarga el viaje para la misma distancia, así que cada kilómetro cuesta más. La salida lo confirma: mediana de 3.52 por km en hora pico contra 3.22 fuera de ella.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La parte de la duración que viene del tráfico. `es_hora_pico` le dice al modelo que ese viaje probablemente será más lento que otro de la misma distancia, que es justo lo que la duración aportaba además de la distancia, y se conoce antes de salir.

**Lo que confirmó la salida, TODO por TODO:**

TODO 1 y 2: hora, día de la semana y fin de semana salen de `fecha_hora_inicio`. TODO 3: tarifa por km mediana de 3.22 fuera de hora pico y 3.52 en hora pico. TODO 4: distancia entre 23 y 0 como número 23, como ciclo 0.26; entre 11 y 12 como número 1, como ciclo 0.26.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Notebook: [`starter/ejercicio2_features_zona.ipynb`](./starter/ejercicio2_features_zona.ipynb), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. La última celda imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes, todos con tarifa mínima y máxima de 70.00. La tarifa de esos viajes no depende de la distancia, pero un modelo lineal solo puede usar la columna para sumar un valor fijo a `a · distancia`: no puede anular el efecto de la distancia en esos viajes. Haría falta una interacción (`es_tarifa_fija × distancia_km`) o un modelo de árboles.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas, una por zona; la ordinal crea 1. Con regresión lineal usaría one-hot. La ordinal asigna números por orden alfabético (Brooklyn 0, Harlem 1, JFK Airport 2, …, Upper West Side 7), y la regresión los trata como cantidades: supone que las zonas tienen un orden y que de Brooklyn a Harlem hay la misma diferencia que de Queens a Upper East Side, lo que es falso.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

`duracion_min` y `propina`, que solo se conocen al terminar el viaje, y `metodo_pago`, que se registra al pagar. `tarifa` es el objetivo. Si las usara para entrenar, el modelo tendría fuga de información: se vería excelente en la evaluación y no podría usarse al pedir el taxi, porque esos datos todavía no existen.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de tu exploración de la Semana 8 y copia la plantilla [`plantilla-features/features_proyecto.ipynb`](./plantilla-features/features_proyecto.ipynb) a la carpeta de tu proyecto. Todo lo que pide este ejercicio queda en el notebook: el código de cada feature, la evidencia y la justificación en las celdas de texto.

### Paso 1: el momento de la predicción (5 minutos)

<<<<<<< HEAD
Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

GrantMatch predice si una convocatoria (award) es pertinente para un investigador, representado por una de sus publicaciones: la probabilidad de que ese award financie ese trabajo. Se predice en el momento de recomendar, cuando la convocatoria está abierta. En ese instante existen los datos del award (funder, país del funder, tipo de financiamiento, campo, monto, duración, año de inicio) y los del perfil del investigador (campo de sus publicaciones, países de sus autores, número de autores, citas). Todavía no existen: si el award lo financió (`label`, que es lo que predecimos), cuántas publicaciones terminó financiando (`funded_outputs_count`) ni las publicaciones que salgan del financiamiento.

Como la OBT solo tiene financiamientos reales (`label` = 1 en todas las filas), el script arma primero pares negativos: a cada award le asigna al azar una publicación que no financió (`label` = 0). Así queda un problema de clasificación con clases balanceadas.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

El código completo está en [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py). Las dos features:

```python
pares["mismo_campo"] = (
    pares["award_field_name"].notna()
    & (pares["award_field_name"] == pares["work_field_name"])
).astype(int)

paises_autores = pares["author_country_codes"].fillna("").astype(str).str.upper().str.split("|")
pais_funder = pares["funder_country_code"].fillna("").astype(str).str.upper()
pares["funder_en_pais_autor"] = [
    int(bool(pais) and pais in paises) for pais, paises in zip(pais_funder, paises_autores)
]
```
=======
En la primera celda de texto, escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

### Paso 2: construye las 2 features (15 minutos)

Crea 2 columnas nuevas, derivadas de las existentes, cada una en su celda y con su evidencia (una correlación, una diferencia de promedios). Al menos una tiene que salir de uno de tus hallazgos de la exploración. Si pruebas una idea que no aporta, deja la celda y escribe por qué la descartas: también es evidencia.
>>>>>>> 11d7186d1047ced2ecbfbbfef422d0dd87851462

### Paso 3: justifica cada feature (5 minutos)

Debajo de cada feature: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta, y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar. Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de features en tu fork:**

<<<<<<< HEAD
`mismo_campo` (1 si el award y la publicación son del mismo campo de investigación). Sale del hallazgo del EDA de la Semana 8: ⟨X⟩ % de los pares reales comparten campo. Evidencia: entre los pares con `mismo_campo` = 1, ⟨X⟩ % son reales, contra ⟨X⟩ % con `mismo_campo` = 0 (correlación con `label` de ⟨X⟩). Existe al predecir: el campo del award viene de la convocatoria y el de la publicación, del perfil del investigador. Es binaria, así que entra tal cual en cualquier modelo.

**Feature 2:**

`funder_en_pais_autor` (1 si el país del funder está entre los países de los autores). Sale del conocimiento del dominio: la mayoría de las agencias financian sobre todo a investigadores de su propio país. Evidencia: entre los pares con `funder_en_pais_autor` = 1, ⟨X⟩ % son reales, contra ⟨X⟩ % con 0 (correlación con `label` de ⟨X⟩). Existe al predecir: el país del funder es parte de la convocatoria y el de los autores, de su afiliación. También es binaria. Si en vez de esto usara el país del funder como categoría, la codificaría con one-hot para un modelo lineal, y ordinal solo para árboles.
=======
_(tu respuesta)_
>>>>>>> 11d7186d1047ced2ecbfbbfef422d0dd87851462

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si uso todos, la media y la desviación ya contienen información de los viajes de prueba, y la evaluación deja de simular viajes nuevos: es fuga de información. En los números cambia poco (la media de distancia pasó de 8.5034 con todos a 8.5576 solo con entrenamiento en la Semana 9.2), pero el procedimiento queda mal, y con datos más chicos o con transformaciones más fuertes la diferencia puede ser grande.
