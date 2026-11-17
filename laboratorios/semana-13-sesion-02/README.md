# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

Porque un modelo se degrada con el tiempo aunque el servicio esté perfecto: los datos de producción cambian y el modelo sigue respondiendo rápido, con código 200, pero cada vez peor. La salud y el volumen dicen si el servicio funciona, no si sus predicciones sirven. Hay que medir además el desempeño funcional del modelo: comparar las predicciones con lo que pasó de verdad (backtesting: precisión, recall o AUC en clasificación, RMSE en regresión), la deriva de los datos y de las predicciones, la calidad de los datos que entran (columnas faltantes, cambios de esquema) y cómo se relaciona todo eso con los indicadores del negocio, separado por versión del modelo.

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

Métricas indirectas (proxies) que se pueden calcular sin la etiqueta: la deriva de los datos (si cambia la distribución de las features de entrada respecto al entrenamiento, muchas veces el primer aviso de que la exactitud va a caer), la deriva de las predicciones (si cambia la distribución de lo que responde el modelo), la deriva de cada feature, la deriva de la importancia de las features (cuánto pesa cada una en las predicciones, que cambia con los reentrenamientos), y la calidad del pipeline de datos. Para medir la deriva propone pruebas como Jensen-Shannon, Kolmogorov-Smirnov y el PSI. Cuando la verdad llega, se guardan las predicciones, se les une la etiqueta y se calcula el desempeño real.

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Que falla en silencio: sigue respondiendo rápido y con 200, pero sus tarifas se alejan de lo que se cobra, porque el mundo cambió (otra tarifa oficial, otro tráfico) y el modelo quedó con lo que vio al entrenar. Ni la latencia ni `/salud` lo ven, porque miden el servicio y no la calidad. Me enteraría guardando cada predicción con un identificador y uniéndola con la tarifa que finalmente se cobró, para calcular el error por semana; y mientras esa verdad llega, vigilando la deriva de los viajes que entran y de las tarifas que responde el modelo.

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

`/metricas` devuelve un diccionario vacío (`{}`) y `desempeno.py` termina con "No hay predicciones.jsonl: el servicio no esta guardando sus predicciones". Lo único que se puede afirmar es que los 1156 viajes recibieron una tarifa: no sé cuánto tardó el servicio, si hubo errores ni si las tarifas se parecieron a lo cobrado. Responder no es lo mismo que funcionar bien, y sin medir no se puede decir ninguna de las dos cosas.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Esperaba una latencia p95 de unos 10 ms (una red pequeña predice una fila en milisegundos) y exactamente 3 errores de validación: las tres peticiones mal formadas que manda `simular_noviembre.py`, sin `fecha_hora_inicio` ni `zona_destino`. No esperaba que la latencia cambiara entre la primera y la última semana: predecir cuesta lo mismo sea cual sea el viaje. Salió p50 8.05 ms, p95 12.14 ms, 1159 peticiones, 3 errores de validación y 0 de servidor.

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

Para unir cada tarifa cobrada con la predicción exacta que hizo el modelo para ese viaje, y así medir el error de cada predicción (y de qué versión). Si solo se guardaran la fecha y la tarifa estimada, no habría cómo emparejarlas: en un mismo minuto hay varios viajes, dos viajes distintos pueden tener la misma tarifa estimada, y una petición repetida o rechazada desordena cualquier unión por posición. El error medido mezclaría viajes, y no se sabría qué modelo se equivocó.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Esperaba un MAE parecido al de prueba, alrededor de 1.17, en las cuatro semanas, porque los viajes de noviembre son del mismo tipo que los de entrenamiento. Salió así solo las dos primeras: 1.18 y 1.34. Desde el 16 de noviembre sube a 4.29 y 4.30, casi 3.7 veces la referencia.

**Lo que mostró el monitoreo:**

Latencia p50 8.05 ms y p95 12.14 ms; 1159 peticiones, 3 errores de validación y 0 de servidor. MAE por semana: 2 de noviembre 1.18 (291 viajes, 1.01 veces la referencia), 9 de noviembre 1.34 (318, 1.15), 16 de noviembre 4.29 (308, 3.67) y 23 de noviembre 4.30 (239, 3.68). Las métricas de sistema estuvieron perfectas todo el mes, mientras el modelo se equivocaba casi cuatro veces más desde el 16.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No hubo votación: en esta sesión no se hizo el ConcepTest en clase.

**Tu razonamiento:**

No aplica (sin ConcepTest en clase).

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica (sin ConcepTest en clase).

## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

De sistema, la de latencia: el 10 de noviembre la p95 fue 78.2 ms (el límite es 50; los demás días, unos 6 ms). De modelo, dos: la indirecta, por la tasa de alertas (la semana del 16 mandó a revisión 0.050 de los pagos y la del 23, 0.046, contra 0.0758 de referencia), y la directa, por el recall (la semana del 16 detectó 0.276 de los fraudes, contra 0.661 de referencia). El día de la latencia alta no coincide con la caída: el 10 de noviembre cae en la semana del 9, que tuvo recall 0.736, normal. El sistema se puso lento un día sin que el modelo empeorara, y el modelo empeoró después sin que el sistema lo notara.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas mostró el problema primero, en la semana del 16 de noviembre: bajó a 0.050 y se pudo ver esa misma semana. El recall lo confirmó recién el 13 de diciembre, cuando esa semana quedó madura (su domingo, 22 de noviembre, más 21 días): 0.276. La tasa de alertas no alcanza porque puede bajar por otras razones: llegaron menos fraudes de verdad (una semana tranquila), cambió la mezcla de pagos o alguien movió el umbral. Una tasa menor con los mismos fraudes es un modelo peor; con menos fraudes, es un modelo igual de bueno. Solo comparando con la verdad se sabe cuál de las dos es.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque esa semana todavía no está madura: su último día es el 29 de noviembre y un fraude puede tardar hasta 21 días en confirmarse, así que pueden seguir llegando contracargos hasta el 20 de diciembre. Si calculara el recall hoy, 14 de diciembre, los fraudes que todavía no se reclamaron contarían como pagos legítimos y el recall saldría distorsionado. Hay que esperar a que la verdad de toda la semana esté completa.

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Con `HOY = "2026-11-30"` solo tiene recall la semana del 2 de noviembre (0.767): es la única cuyo domingo más 21 días (29 de noviembre) ya pasó. Las alertas que salen son la de latencia del 10 y las dos indirectas de la tasa de alertas, pero todavía no la del recall. La caída del recall de la semana del 16 se puede mostrar desde el 13 de diciembre (domingo 22 más 21 días). Es decir, la tasa de alertas avisó casi un mes antes que el recall.

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

Mucho: meses o años. En GrantMatch un par (award, investigador) se confirma cuando una publicación del investigador aparece en OpenAlex reconociendo ese award, y eso pasa después del financiamiento y de la publicación, y otra vez después, cuando OpenAlex la enlaza. `monitoreo/desempeno.py` solo calcula la AUC y el recall de las semanas a las que ya pasaron 365 días. Mientras tanto avisan las métricas que se conocen el mismo día: la tasa de recomendación y la probabilidad promedio (deriva de las predicciones), y la calidad de la entrada, sobre todo la fracción de pares en que el país del financiador coincide con el de un autor, que es la feature que más pesa. Dicen si el modelo recibe y responde lo mismo que en la validación, no si acierta.

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Que recomiende mal sin fallar. Si OpenAlex cambia cómo publica los países de los autores o deja de traer el país del financiador, `funder_en_pais_autor` sale 0 para todos, y el modelo responde rápido, con 200, pero recomienda casi al azar. Lo mismo si cambia la forma en que las agencias financian (más colaboración internacional) y la relación entre país y financiamiento se debilita: es deriva de concepto, y la entrada se ve igual. Ninguna métrica de sistema (latencia, errores, peticiones) lo ve: solo las de modelo, la deriva de esa feature y de la tasa de recomendación el mismo día, y la AUC y el recall cuando llega la verdad.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió la relación entre el viaje y su tarifa, no los viajes: es deriva de concepto. Con viajes de la misma distancia (8.8 km de promedio la semana del 16 contra 8.5 en octubre), la tarifa cobrada subió unos 4.2 dólares por viaje (31.41 contra 27.18 estimados), como si hubiera entrado una tarifa nueva o un recargo fijo. No lo hubiera detectado mirando solo los viajes que llegan: su distribución es la misma, así que cualquier prueba de deriva de datos (PSI, KS sobre distancia, zonas u hora) habría salido tranquila, y la deriva de las predicciones también, porque el modelo sigue estimando lo mismo. Solo se ve al unir cada predicción con la tarifa cobrada, que es lo que hace `desempeno.py`. La lectura previa ya no está en línea (WhyLabs cerró y su blog redirige a un aviso), así que la respuesta sale del concepto, no del artículo.
