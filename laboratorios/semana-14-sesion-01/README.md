# Semana 14, Sesión 1: Monitoreo en producción II: deriva de datos y deriva conceptual

> Objetivo de la sesión: distinguir deriva de datos de deriva conceptual en el modelo de tu proyecto, y diseñar una alerta que detecte cuál de las dos lo está degradando.

**Lectura previa de esta sesión:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 9: Continual Learning and Test in Production.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El Ejercicio 1 es un notebook (simula la deriva y elige el umbral); el Ejercicio 2, un script del servicio (la alerta queda ahí, corriendo con cada predicción).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la deriva de datos y qué es la deriva conceptual? Da un ejemplo de cada una.**

La deriva de datos es un cambio en la distribución de los datos de entrada que recibe el modelo, P(X): llegan otros casos que los de entrenamiento. Ejemplo: en una semana de feriado se piden muchos más viajes al aeropuerto. La deriva conceptual es un cambio en la relación entre las entradas y lo que el modelo predice, P(y | X): los mismos casos ahora tienen otra respuesta correcta. Ejemplo: la ciudad sube 20 % la tarifa por taxímetro, y un viaje idéntico al de octubre ahora cuesta más. Las dos pueden venir de cambios en el entorno, en el comportamiento de los usuarios o en las fuentes de datos. (Nota: la lectura ya no está en línea, porque WhyLabs cerró y su blog redirige a un aviso; la respuesta sale del resumen del artículo y de la definición estándar.)

**¿Por qué el artículo dice que hay que monitorear las dos, y no solo el desempeño del modelo?**

Porque las dos degradan la exactitud del modelo con el tiempo, pero de formas distintas, y el desempeño solo no alcanza: muchas veces la verdad llega tarde (la tarifa cobrada, un contracargo, una publicación), así que la deriva de datos sirve de aviso temprano mientras tanto. Y cuando el desempeño cae, saber cuál de las dos derivas hay dice qué hacer: si cambiaron los datos, revisar el pipeline o reentrenar con más casos del nuevo tipo; si cambió la relación, reentrenar con etiquetas recientes, porque las viejas ya no valen. Además, no toda deriva de datos empeora el modelo, y una deriva conceptual puede no verse en los datos: hay que mirar las dos cosas.

## Pregunta de apertura

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió la relación entre el viaje y su tarifa, no los viajes: es deriva conceptual. Con viajes de la misma distancia (8.8 km de promedio la semana del 16, contra 8.5 en octubre), la tarifa cobrada subió unos 4.2 dólares por viaje (31.41 contra 27.18 estimados), como si hubiera entrado una tarifa nueva. No lo hubiera detectado mirando solo los viajes que llegan: su distribución es la misma, así que la deriva de datos (el PSI de distancia, zona u hora) sale tranquila, y las predicciones tampoco cambian, porque el modelo estima lo mismo que antes. Solo se ve al unir cada predicción con la tarifa cobrada.

## Ejercicio 1: ¿deriva de datos o deriva conceptual?

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`ejercicio1_deriva.ipynb`](./starter/ejercicio1/ejercicio1_deriva.ipynb) entrena la versión 1 del modelo de la tarifa y simula semanas de 300 viajes: semanas normales, semanas de feriado con más viajes al aeropuerto (deriva de datos) y semanas con un aumento de la tarifa por taxímetro (deriva conceptual). Para cada una mide el PSI de dos features contra los viajes de entrenamiento y el MAE del modelo. Los 4 `TODO`: la fórmula del PSI, medir 200 semanas normales, y las dos derivas simuladas.

Corre todas las celdas antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1 y 2):** antes de medir las semanas normales, ¿qué PSI esperas para una semana en la que nada cambió? Después de medirlas: ¿cuál fue el máximo en 200 semanas?

Esperaba un PSI cercano a 0 pero no exactamente 0: una muestra de 300 viajes nunca reparte igual que la referencia, solo por azar. En 200 semanas normales el PSI de distancia tuvo promedio 0.050 y máximo 0.121, y el de zona promedio 0.035 y máximo 0.109 (percentil 95: 0.092 y 0.080). El MAE se movió entre 1.16 de promedio y 1.32 como máximo.

**Respuesta 2 (TODO 3 y 4):** antes de correr, ¿qué esperas que pase con el PSI y con el MAE en la semana de feriado? ¿Y en la semana del aumento?

En el feriado esperaba que el PSI subiera mucho (cambian las distancias y las zonas) y que el MAE casi no cambiara, porque el modelo vio viajes al aeropuerto al entrenar y muchos tienen tarifa fija. En el aumento esperaba lo contrario: el PSI igual al de una semana normal, porque los viajes son los mismos, y el MAE muy alto, porque la tarifa real es 20 % mayor de lo que el modelo aprendió. Fue así: con el aeropuerto ×4, PSI 0.19 y 0.20 y MAE 0.97; con el aumento de 20 %, PSI 0.08 y 0.06 y MAE 4.16.

**Respuesta 3 (Paso 4):** en la semana de feriado con peso 8, el PSI es más de diez veces el de una semana normal. ¿Empeoró el modelo? ¿Por qué?

No. Con peso 8, el PSI de distancia es 0.54 y el de zona 0.42 (diez veces una semana normal), pero el MAE baja a 0.94, mejor que el 1.19 de una semana normal. Los viajes al aeropuerto con Manhattan tienen tarifa fija (70 dólares), y el modelo aprendió a acertarla casi exacto; si llegan más de esos, su error promedio baja. Que los datos cambien no significa que el modelo empeore: depende de si el cambio lleva a zonas donde el modelo es malo o bueno.

**Respuesta 4 (Paso 5):** ¿qué umbral de PSI elegirías para la alerta, y con qué dos números de la tabla lo justificas?

0.15. En las 200 semanas normales no da ninguna falsa alarma (0.0 %; con 0.10 serían 4.0 %, porque el máximo normal fue 0.121), y detecta el 97.5 % de las semanas de feriado con el aeropuerto ×4. Con 0.20 la detección baja a 71.5 %, y con 0.10 se ganaría poco en detección (100 %) a cambio de una falsa alarma cada 25 semanas.

**Lo que mostró la simulación:**

| Escenario | PSI de distancia | PSI de zona | MAE |
|---|---|---|---|
| Semanas normales (máximo de 200) | 0.121 | 0.109 | 1.32 |
| Feriado, aeropuerto ×4 | 0.221 | 0.197 | 1.04 |
| Aumento de 20% | 0.046 | 0.043 | 4.05 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No hubo votación: en esta sesión no se hizo el ConcepTest en clase.

**Tu razonamiento:**

No aplica (sin ConcepTest en clase).

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica (sin ConcepTest en clase).

## Ejercicio 2: la alerta, en el servicio

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de la tarifa de la sesión pasada tiene ahora una ruta `/deriva`. [`deriva.py`](./starter/ejercicio2/deriva.py) trae `MonitorDeDeriva`: guarda los últimos 300 viajes que recibió el servicio y calcula el PSI de `distancia_km` y de `zona_origen` contra la referencia que guarda `entrenar.py` (`referencia_deriva.json`), con el umbral de 0.15. Los pasos 1 a 3 tienen `TODO` (las proporciones numéricas, las categóricas y el PSI); los pasos 4 (guardar solo los últimos viajes) y 5 (el estado, con la alerta) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`. Después:

```
python entrenar.py
python simular_semanas.py
```

[`simular_semanas.py`](./starter/ejercicio2/simular_semanas.py) manda al servicio una semana normal, una de feriado y una del aumento, y muestra `/deriva` después de cada una (no lo modifiques).

**Respuesta 1:** ¿qué dijo `/deriva` después de cada semana? ¿Alguna alerta te sorprendió, o te faltó?

Después de la semana normal: PSI de distancia 0.053 y de zona 0.055, sin alerta. Después del feriado: 0.258 y 0.221, con alerta. Después del aumento: 0.058 y 0.048, sin alerta. Ninguna me sorprendió: era lo que mostró el notebook. Lo que falta es una alerta en la semana del aumento, que es la que más daña al modelo, pero esa no la puede ver un monitor que solo mira los viajes que llegan.

**Respuesta 2:** la alerta del monitor no dijo nada en la semana del aumento, que es la que más empeora el modelo. ¿Qué alerta de la sesión pasada sí la habría visto, y por qué tarda más?

La del MAE por semana de `desempeno.py`, que une cada predicción con la tarifa cobrada: ve que el error se multiplica por casi cuatro. Tarda más porque necesita la verdad: la tarifa cobrada llega cuando termina el viaje, y para que el MAE de una semana sea confiable hay que esperar a juntar los viajes de esa semana. El PSI se calcula con lo que pide la app, en el momento.

**Respuesta 3:** escribe la regla que usaría el equipo para decidir, con las dos alertas (la de deriva de este ejercicio y la del MAE de la sesión pasada), si lo que pasa es deriva de datos, deriva conceptual o nada. ¿Qué haría en cada caso?

Si sube el PSI y el MAE no: deriva de datos que el modelo aguanta. Revisar que no sea un error del pipeline (una columna rota, una app vieja), anotarlo y seguir vigilando el MAE; no hace falta reentrenar. Si sube el PSI y también el MAE: deriva de datos que daña al modelo. Reentrenar con datos recientes que incluyan los casos nuevos, y promover la versión solo si baja el MAE. Si sube el MAE y el PSI no: deriva conceptual, cambió la relación (por ejemplo, la tarifa). Investigar la causa con el negocio, reentrenar con etiquetas posteriores al cambio (las anteriores ya no valen) y, mientras tanto, si se conoce el cambio (un aumento de 20 %), corregir o volver a la regla manual. Si no sube ninguno: nada que hacer.

**Si terminas antes:** el monitor mira solo dos features. Inventa una semana con deriva de datos que este monitor no vería. ¿Empeoraría el modelo?

Una semana con muchos más pasajeros por viaje (grupos de turistas) o con otro reparto de métodos de pago: el monitor solo mira `distancia_km` y `zona_origen`, así que no la vería. Probablemente no empeoraría el modelo, porque en estos datos la tarifa depende sobre todo de la distancia, la zona y la tarifa fija, y poco de los pasajeros o del pago. Una que sí lo dañaría sin que el monitor la viera: más viajes en hora pico (más tráfico, más tiempo de taxímetro) con las mismas distancias y zonas. Para eso habría que vigilar también la hora.

## Ejercicio 3: la deriva de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio instrumentado de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: la simulación, en `monitoreo/`

Un notebook o un script que, con tus datos de prueba:

- mida el PSI de al menos dos de tus features en ventanas normales (del tamaño de lo que recibe tu servicio en un período), para saber cuánto se mueve sin que nada cambie;
- simule una deriva de datos y una deriva conceptual que tengan sentido en tu problema, y mida el PSI y tu métrica de la Semana 11 en cada una;
- elija el umbral de la alerta con esos números, y guarde la referencia que necesita el servicio.

### Entregable 2: la alerta, en el servicio

Un monitor de deriva en tu servicio (puedes adaptar `deriva.py`) con una ruta que diga el PSI de la ventana actual y si hay alerta, y una prueba en `servicio/test_api.py` que compruebe que la alerta se enciende con una ventana alejada de la referencia.

### Entregable 3: `docs/monitoreo.md`

Agrega la sección **Deriva**: la tabla de la simulación, el umbral elegido y por qué, y qué deriva no detecta tu alerta.

**En tu proyecto, ¿qué deriva de datos es la más probable, y empeoraría tu modelo? ¿Y qué deriva conceptual?**

La de datos más probable es que cambie la mezcla de financiadores y de publicaciones: en mis propios datos ya pasó, de los awards anteriores a 2020 a los de 2020 en adelante los financiadores de China bajan del 17 % al 7.5 % de los pares (PSI 0.27) y las publicaciones nuevas tienen menos citas (PSI 0.14), y aun así el modelo da AUC 0.90. Simulando muchas más convocatorias de China (×4 y ×8), el PSI de país sube a 0.25 y 0.70 y la AUC apenas baja (0.895 y 0.886): no empeora mucho. La conceptual más probable es que los financiadores empiecen a financiar a autores de otros países (programas internacionales): la feature que más pesa, `funder_en_pais_autor`, dejaría de significar lo mismo. Con 25 % y 50 % de financiamiento internacional la AUC cae a 0.68 y 0.46 y el PSI no se mueve: esa sí empeora el modelo, y mucho.

**Si tu alerta de deriva se enciende y tu métrica de desempeño no cambia, ¿qué haces?**

Primero reviso que no sea un error del pipeline: una columna que dejó de venir de OpenAlex o un cambio en la OBT, que es lo más común y se arregla sin tocar el modelo. Si los datos son buenos, es deriva de datos que el modelo aguanta (como China ×8, con AUC 0.886): lo anoto, sigo vigilando, y si se mantiene muevo la referencia de deriva al período nuevo para que la alerta no se quede encendida. No reentreno solo por el PSI. Igual hay que tener en cuenta que en GrantMatch la AUC llega un año tarde: que no haya cambiado todavía puede ser solo que la verdad no llegó, así que miro también la tasa de recomendación y la probabilidad promedio.

## Antes de la próxima sesión

**Lectura:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

En las últimas cuatro sesiones construiste un servicio, un registro, métricas y alertas, con PSI, MAE, p95 y recall. ¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

Al científico de datos le explico cómo funciona y qué tan confiable es: el modelo, la validación, las métricas y sus límites. A quien paga le explico qué gana y qué riesgo corre, en su idioma (dinero, tiempo, investigadores que no se pierden una convocatoria), sin PSI ni AUC, con una recomendación concreta y lo que necesito de él. Al científico de datos: "Es un gradient boosting sobre pares award–publicación de OpenAlex, con AUC de 0.90 en validación por award, servido por lotes con registro de versiones en MLflow y alertas de deriva por PSI; su debilidad es que casi toda la señal viene de si el financiador es del país de algún autor." A quien paga: "GrantMatch le muestra a cada investigador las convocatorias que le corresponden y encuentra 98 de cada 100 que de verdad son para él, a cambio de que descarte algunas que no; y si deja de funcionar bien, nos avisa antes de que alguien lo note."
