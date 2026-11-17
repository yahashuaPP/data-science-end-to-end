# Semana 14, Sesión 2: Comunicación técnica y ejecutiva de los resultados

> Objetivo de la sesión: adaptar la presentación de los resultados y limitaciones de tu proyecto a una audiencia técnica y a una audiencia ejecutiva.

**Lectura previa de esta sesión:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 11: The Human Side of Machine Learning.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Hoy se escribe más de lo que se programa: los dos scripts preparan los números y los gráficos, y el trabajo es decidir qué se dice a quién.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué hay que saber de la audiencia antes de preparar la presentación? ¿Qué recomienda hacer con la jerga técnica?**

Quién es, qué sabe y qué decisión tiene que tomar con lo que le cuentes: su nivel técnico, qué le preocupa (dinero, riesgo, tiempo, clientes) y qué necesita para actuar. Con eso se elige qué resultados mostrar, en qué unidades y con cuánto detalle, y se empieza por la conclusión y lo que significa para ella, no por el método. Con la jerga técnica: evitarla o traducirla al idioma del negocio (en vez de "MAE de 4.29", "le cobramos en promedio 4 dólares más de lo que le mostramos"), y si un término es inevitable, explicarlo en una frase. (Nota: no pude abrir el artículo desde aquí, así que esta respuesta no sale de su texto; revísala contra lo que leíste.)

**¿Qué lugar le da el artículo a las limitaciones y a la incertidumbre de los resultados?**

Un lugar explícito y honesto, no escondido al final: decir qué no hace el modelo, con qué datos se probó y cuánta incertidumbre tienen los números, en el lenguaje de la audiencia (un rango, "en 9 de cada 10 casos"), para que la decisión se tome sabiendo el riesgo. Callarlas genera más confianza de la debida, y cuando aparecen se pierde la credibilidad. (Igual que la anterior, sin el texto del artículo a la vista.)

## Pregunta de apertura

¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

Para un científico de datos: "Es un gradient boosting sobre pares award–publicación de OpenAlex, con AUC 0.90 en validación por award contra 0.5 de la línea base, servido por lotes, con registro de versiones en MLflow y alertas de deriva por PSI; su punto débil es que casi toda la señal viene de si el financiador es del país de algún autor."

Para quien paga: "GrantMatch le muestra a cada investigador 98 de cada 100 convocatorias que de verdad son para él, a cambio de que descarte algunas que no le sirven, y nos avisa si deja de funcionar bien."

## Ejercicio 1: noviembre, para dos audiencias

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`tarifas_noviembre.csv`](./starter/ejercicio1/tarifas_noviembre.csv) tiene los 1156 viajes de noviembre del monitoreo de la Semana 13: la tarifa estimada por la versión 1, la cobrada y si el viaje tenía tarifa fija. [`graficos_tarifa.py`](./starter/ejercicio1/graficos_tarifa.py) calcula el resumen por semana y dibuja dos gráficos con los mismos datos, `grafico_tecnico.png` y `grafico_ejecutivo.png`. Los 3 `TODO`: el resumen, el gráfico técnico y el gráfico ejecutivo.

```
python graficos_tarifa.py
```

**Respuesta 1 (TODO 1):** el MAE pasó de 1.18 a 4.29. Antes de calcularlo: ¿qué número de la tabla le dirías a la gerente de operaciones, que nunca oyó hablar del MAE?

La fracción de pasajeros que pagan más de 20 % sobre la tarifa que les mostró la app: pasó de 3 % y 6 % en las dos primeras semanas a 47 % y 46 % desde el 16 de noviembre. Casi la mitad de los pasajeros paga bastante más de lo que se le prometió. También le diría la diferencia en dólares: en promedio, 4.20 dólares más por viaje (antes, 0.16 y −0.07).

**Respuesta 2 (TODO 2 y 3):** los dos gráficos usan los mismos viajes. Haz una lista de todo lo que cambia entre uno y otro: la métrica, el título, la cantidad de series, los colores.

La métrica: MAE en dólares contra porcentaje de pasajeros que pagan más de 20 % sobre lo estimado. El título: describe lo medido ("MAE semanal de la versión 1, por tipo de tarifa") contra la conclusión ("Desde el 16/11, casi la mitad de los pasajeros paga más de 20 % sobre la tarifa que le mostramos"). Las series: dos (taxímetro y tarifa fija) más una línea de referencia (el MAE de prueba, 1.17), contra una sola serie sin referencia técnica. Los colores: uno por serie, contra gris para las semanas normales y rojo solo para las dos semanas con el problema. Las etiquetas: el ejecutivo pone el valor encima de cada barra y quita los bordes y el detalle; el técnico tiene leyenda, unidad en el eje (dólares) y la semana como "lunes".

**Respuesta 3:** escribe las dos diapositivas. Cada una tiene un título que es una oración completa, el gráfico y como máximo cuatro líneas.

**Diapositiva técnica** (para el equipo de datos):

**La versión 1 se equivoca 3.7 veces más desde el 16/11 solo en tarifas por taxímetro: es deriva conceptual, no de datos.**

- Se midió el MAE semanal uniendo cada predicción con la tarifa cobrada (1 156 viajes de noviembre).
- Taxímetro: de 1.30 y 1.41 a 4.75 y 4.62 dólares; tarifa fija: estable (0.22–0.44). La diferencia media pasa de 0.16 a +4.23: lo cobrado sube, no el error al azar.
- Con las mismas distancias y zonas que en octubre: cambió P(tarifa | viaje). Hay que reentrenar con tarifas posteriores al 16/11.
- Limitación: 4 semanas, ~300 viajes cada una; la causa (¿un aumento del taxímetro?) no está en los datos.

**Diapositiva ejecutiva** (para la gerente de operaciones):

**Desde el 16 de noviembre, casi la mitad de los pasajeros paga más de 20 % sobre la tarifa que le mostramos.**

- Qué pasó: la tarifa que cobramos subió unos 4 dólares por viaje en taxímetro y la app no se enteró; los viajes al aeropuerto con tarifa fija están bien.
- Qué cuesta: 47 % de los pasajeros paga más de 20 % sobre lo prometido (antes, 3 %): reclamos y pérdida de confianza.
- Qué recomendamos: actualizar el modelo con las tarifas de noviembre esta semana.
- Qué necesitamos de usted: confirmar si hubo un cambio oficial en la tarifa del taxímetro y desde qué fecha.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No hubo votación: en esta sesión no se hizo el ConcepTest en clase.

**Tu razonamiento:**

No aplica (sin ConcepTest en clase).

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica (sin ConcepTest en clase).

## Ejercicio 2: el fraude, en dólares

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El reporte técnico de la Semana 13 decía que el recall de la semana del 16 cayó de 0.74 a 0.28. [`costos_fraude.py`](./starter/ejercicio2/costos_fraude.py) lo traduce a dólares con los costos de la Semana 11 (40 por fraude no detectado, 3 por revisar un pago legítimo). Los pasos 1 a 3 tienen `TODO`: los errores por semana, su costo y el costo de una semana normal. Los pasos 4 (la oración para la gerencia) y 5 (lo que cuesta un mes así) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python costos_fraude.py`.

**Respuesta 1:** ¿qué oración te dio el paso 4? ¿Qué le falta para que la gerente pueda decidir algo?

"La semana del 2026-11-16, el fraude que no detectamos y las revisiones costaron 2359 dolares, 2.7 veces una semana normal." Le falta lo necesario para decidir: por qué pasó (el modelo dejó de reconocer un patrón nuevo de fraude), qué se propone hacer y cuánto cuesta hacerlo, qué se pierde si no se hace nada (unos 5 968 dólares más por mes si siguiera así) y qué se le pide a ella (aprobar algo, asignar a alguien, un plazo).

**Respuesta 2:** escribe el mensaje ejecutivo completo, en cinco oraciones como máximo: qué pasó, cuánto cuesta, qué recomiendan, qué necesitan y qué riesgo queda.

Desde la semana del 16 de noviembre, el modelo de fraude dejó pasar casi 3 de cada 4 fraudes, y esa semana el fraude no detectado más las revisiones costaron 2 359 dólares, 2.7 veces una semana normal (867). Si sigue así, un mes nos cuesta unos 6 000 dólares más de lo normal. Recomendamos reentrenar el modelo esta semana con los fraudes confirmados de noviembre y, mientras tanto, bajar el umbral para mandar más pagos a revisión. Necesitamos que apruebe unas horas más del equipo de revisión durante dos semanas. El riesgo que queda es que el fraude de las dos últimas semanas de noviembre todavía no se confirma del todo, así que el costo real puede ser mayor.

**Respuesta 3:** el costo de 40 dólares por fraude no detectado es una suposición de la Semana 11. ¿Dónde lo dirías, en el mensaje ejecutivo o en el técnico? ¿Qué pasa con tu recomendación si el costo real fuera la mitad?

En el técnico, con el detalle de cómo se calculó; en el ejecutivo basta una mención corta ("con un costo estimado de 40 dólares por fraude"), para que la gerente sepa que es una estimación. Si el costo real fuera la mitad (20 dólares), la semana del 16 costaría 1 100 + 159 = 1 259 dólares contra unos 507 de una semana normal (360 de fraude más 147 de revisiones): sigue siendo 2.5 veces, así que la recomendación no cambia (reentrenar sigue valiendo la pena), pero baja la urgencia de gastar en más revisores: con el fraude más barato, mandar más pagos a revisión rinde menos.

## Ejercicio 3: los resultados de tu proyecto, dos veces

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/resultados_tecnico.pdf`

Una diapositiva para otra persona del equipo de datos: el modelo y la línea base, la métrica de la Semana 11 con su esquema de validación, lo que mostró el monitoreo y las limitaciones, con los números.

### Entregable 2: `docs/resultados_ejecutivo.pdf`

La misma diapositiva para quien decide si se usa el modelo: un título que dice la conclusión, el resultado en la unidad de quien decide (dinero, tiempo, casos), qué recomiendan, qué necesitan y qué no hace el modelo. Sin el nombre del algoritmo ni de la métrica.

Puedes armar las dos diapositivas en lo que prefieras (Google Slides, PowerPoint, Markdown exportado) y guardarlas en PDF.

**¿Qué número de tu proyecto usaste en la versión ejecutiva que no aparece en la técnica? ¿De dónde salió?**

"1 065 convocatorias pertinentes más que llegan al investigador": la diferencia entre las que se pierden con el umbral de 0.5 (1 209) y con el de 0.10 (144), de las 8 220 pertinentes de la validación cruzada por award (`validacion_proyecto.ipynb`). La técnica dice lo mismo como "recall 0.982, costo −46 %". También usé "562 mil USD", la mediana de `amount_usd` de la muestra de la OBT, y "~7 de 10" (5 891 recomendaciones de más por cada 8 076 acertadas), que salen de la misma validación.

**¿Qué limitación dejaste en las dos versiones, y cómo la dijiste en cada una?**

Que el modelo está probado con una parte de los datos: en la técnica, "solo el 16 % de los awards trae año de inicio: entrenado con 16 440 pares, sobre todo NSF, NSFC, JSPS y Comisión Europea"; en la ejecutiva, "está probado sobre todo con grandes agencias (NSF, China, Japón, Unión Europea)". Y que la deriva conceptual tarda en verse: en la técnica, "AUC 0.46 con PSI normal; solo la vería la AUC con la verdad, que tarda ~1 año"; en la ejecutiva, "si los financiadores cambian a quién financian, tardamos hasta un año en confirmarlo".

## Antes de la próxima sesión

**Lectura:** ["AI Model Lifecycle Management: From Development to Decommissioning"](https://blog.cognitiveview.com/ai-model-lifecycle-management-from-development-to-decommissioning/), CognitiveView.

Hoy escribiste "qué no hace el modelo" en una diapositiva. Ahora imagina que sí lo hace mal: ¿bajo qué condición apagarías el modelo de tu proyecto, y quién tendría que enterarse?

Lo apagaría (volvería a una lista sin modelo, por ejemplo convocatorias del mismo dominio ordenadas por fecha de cierre) si la AUC por award de una semana madura cae por debajo de 0.75, si el recall al umbral de 0.10 cae por debajo de 0.85 durante dos semanas maduras seguidas, o si el pipeline deja de traer el país del financiador o de los autores (la feature que más pesa saldría 0 para todos y el modelo recomendaría casi al azar). También si el negocio cambia: por ejemplo, si GrantMatch empieza a recomendar convocatorias de agencias con las que no se probó. Tendrían que enterarse el equipo de datos (para reentrenar o retirar la versión en el registro, moviendo el alias), quien es dueño del producto (decide qué ven los investigadores mientras tanto) y los investigadores, con un aviso de que la lista se ordena de otra forma por un tiempo.
