# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los laboratorios de hoy son notebooks de Jupyter, como el diagnóstico de la Semana 4: cada gráfico aparece debajo de la celda que lo dibuja. Una exploración se lee para decidir qué hacer con los datos; por eso va en un notebook y no en un script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

La pregunta de fondo es "¿con qué estoy trabajando?": cuáles son las características de los datos, qué anomalías tienen y qué patrones y relaciones esconden, antes de preprocesar y entrenar. La autora (Miriam Santos) la divide en tres tareas: un panorama del conjunto (cuántas observaciones, de qué tipo es cada variable, cuántos faltantes y duplicados, con estadística descriptiva), la evaluación de cada variable y de cómo se relacionan (análisis univariado y multivariado), y la calidad de los datos (faltantes, desbalance, valores constantes o duplicados, variables redundantes o ruidosas). Además es el punto de partida de las hipótesis que después se prueban, y no se hace una sola vez: se repite en cada etapa del preprocesamiento y del modelado.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una sola variable: si es numérica, estadística descriptiva (media, desviación estándar, asimetría, curtosis, cuantiles) y un histograma; si es categórica, la moda y una tabla de frecuencias, con un gráfico de barras. Para la relación entre dos: gráficos de interacción entre pares de variables y coeficientes de correlación en una matriz o mapa de calor, eligiendo el coeficiente según el tipo de variable (Spearman entre numéricas, V de Cramér entre categóricas o entre una categórica y una numérica discretizada). Todo eso lo junta en una herramienta, ydata-profiling, que con un solo `ProfileReport` da el panorama, las dos exploraciones, las correlaciones y alertas automáticas de calidad (duplicados, alta correlación, desbalance, faltantes, ceros).

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo intenta ajustar esos 15 viajes, que están a casi 1000 dólares de todo lo demás: en una regresión lineal arrastran los coeficientes y empeoran la predicción de los 2985 viajes normales; en un árbol crean hojas que predicen tarifas absurdas. Mirando solo el resultado final probablemente no lo notaría: el error promedio sube, pero no dice por qué. Se descubre mirando los datos antes de entrenar (un `describe()` o un histograma).

## Ejercicio 1: resumen, distribución y atípicos

Notebook: [`starter/ejercicio1_explorar.ipynb`](./starter/ejercicio1_explorar.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Ya lee el archivo y arma los gráficos; faltan 4 `TODO`. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El máximo de `tarifa` (999.99), que no es una tarifa plausible para un taxi y está muy lejos del percentil 75 (32.55), y el mínimo de `duracion_min` (-74), porque una duración no puede ser negativa. El mínimo de `distancia_km` en 0 también es sospechoso.

**Respuesta 2 (TODO 2):** mira el histograma después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Porque los 15 viajes con tarifa de 999.99 estiran el eje x hasta 1000. Las otras 2985 tarifas están por debajo de 78, así que ocupan menos de una décima del ancho del gráfico y quedan aplastadas a la izquierda.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No. El IQR solo dice que un valor está lejos del centro, no que esté mal. Lo averiguo mirando qué tarifas concretas son atípicas (`value_counts`) y de qué viajes son: si un valor exacto se repite cientos de veces y siempre en las mismas zonas, es una regla de negocio y no un error de captura.

**Lo que confirmó la salida, TODO por TODO:**

Q1 = 13.99, Q3 = 32.55, IQR = 18.56, límite superior = 60.40. El IQR marcó 334 viajes: 302 con tarifa exacta de 70.00 (tarifa fija del aeropuerto), 15 con 999.99 (error de captura) y 17 viajes largos con tarifas entre 61.07 y 77.81. Aparte, hay 10 duraciones negativas que el IQR de la tarifa no detecta.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No hubo votación: en esta sesión no se hizo el ConcepTest en clase.

**Tu razonamiento:**

No aplica (sin ConcepTest en clase).

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica (sin ConcepTest en clase).

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Notebook: [`starter/ejercicio2_relaciones.ipynb`](./starter/ejercicio2_relaciones.ipynb), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. La última celda imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la tendencia lineal (más distancia, más tarifa) hay dos grupos que no la siguen. Una línea horizontal en 70: 300 viajes que cuestan exactamente 70 sin importar la distancia, y todos van entre JFK Airport y una zona de Manhattan (tarifa fija). Y una línea vertical en distancia 0: 25 viajes con tarifas reales, en los que la distancia no se registró.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

En efectivo la propina es 0 en los 888 viajes (promedio 0.00); con tarjeta el promedio es 4.73 y ningún viaje tiene propina 0. Que sea exactamente 100 % contra 0 % describe al sistema, no a los pasajeros: la propina en efectivo no se registra. Además hay 60 viajes sin método de pago.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración solo se conoce cuando el viaje termina, así que no existe en el momento de pedir el taxi. Su correlación con la tarifa (0.956) es casi la de la distancia (0.960) porque las dos crecen juntas, pero usarla sería fuga de información: el modelo se vería muy bien al evaluarlo y no podría usarse en producción. Lo que sí está disponible es la distancia estimada y la hora.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar. Copia la plantilla [`plantilla-eda/explorar_proyecto.ipynb`](./plantilla-eda/explorar_proyecto.ipynb) a `psets/proyecto/modelado/`: el notebook es tu entregable, con el código, lo que mostró y lo que concluyes en el mismo lugar.

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3: filas, columnas, faltantes por columna, la distribución del objetivo y cuántos atípicos marca el IQR.

### Paso 2: relaciones (10 minutos)

Completa el paso 4: la relación entre tu objetivo y la variable que esperas que más influya.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

En el paso 5 del notebook, para cada hallazgo: una celda de código con el número o el gráfico que lo confirma, y debajo, en la celda de texto, qué encontraste y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar). Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de exploración en tu fork:**

[`psets/proyecto/modelado/explorar_proyecto.ipynb`](../../psets/proyecto/modelado/explorar_proyecto.ipynb), sobre la muestra de la OBT que exporta [`psets/proyecto/modelado/exportar_obt.sql`](../../psets/proyecto/modelado/exportar_obt.sql).

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Creando columnas nuevas a partir de las que ya existen: una columna `es_tarifa_fija` que vale 1 cuando el viaje va entre JFK y una zona de Manhattan, y, de `fecha_hora_inicio`, la hora, si es hora pico y la hora como ciclo (seno y coseno), para que 23 y 0 queden cerca.
