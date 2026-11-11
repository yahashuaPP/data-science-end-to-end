# Despliegue del modelo de GrantMatch: por lotes, cada noche

**Decisión:** el modelo se sirve **por lotes**. Una vez por noche, después de que el pipeline de datos (Kestra + dbt) actualiza la OBT en Snowflake, un proceso puntúa los pares (award abierto, investigador) y deja la probabilidad de financiamiento en una tabla. La aplicación lee de esa tabla. El servicio de `servicio/` existe para lo que el lote no cubre (ver la última sección), no como el camino principal.

**El número que lo decide: 24 horas.** Es lo que puede esperar la recomendación sin que nadie pierda nada.

## Qué decisión habilita la predicción, quién la usa y cuánto puede esperar

La predicción decide **qué convocatorias se le muestran a cada investigador**, ordenadas por la probabilidad de que el award financie un trabajo como el suyo. La usa el investigador (en su lista de convocatorias recomendadas o en un correo semanal), y de forma agregada quien analiza a qué campos y países llega el financiamiento.

Una convocatoria está abierta semanas o meses, y preparar una postulación toma días. Que una convocatoria publicada hoy aparezca en la lista mañana en vez de dentro de 200 milisegundos no cambia ninguna decisión del investigador. **Tiempo de espera aceptable: hasta un día.** No hay ninguna persona mirando una pantalla mientras el modelo responde.

## Cuántos casos hay que predecir, y si se conocen de antemano

Los casos son pares (award, investigador). Los dos lados se conocen antes de que nadie pregunte:

- Los awards salen de la OBT (`OBT.OBT_AWARD_WORK`), que el pipeline del equipo carga desde OpenAlex. Para recomendar solo interesan los que siguen abiertos.
- Los investigadores son los autores con publicaciones en la misma OBT: su perfil (campo, país, citas) es el de sus publicaciones.

El número de pares es (awards abiertos) × (investigadores): ⟨número de awards en la muestra⟩ × ⟨número de autores⟩. Es grande, pero finito, y conocido cada noche. Además se puede recortar antes de puntuar: comparar solo pares del mismo dominio (`award_domain_name` = `work_domain_name`) descarta la mayoría de las combinaciones sin perder casi ningún par real. Los modelos de scikit-learn que comparé (logística, bosques, gradient boosting) puntúan cientos de miles de filas por minuto en una laptop, así que el lote entero cabe con holgura en la noche.

## Qué datos necesitan las features en el momento de predecir

| Feature | De dónde sale | ¿Disponible al predecir? |
|---|---|---|
| `amount_usd`, `duration_days`, `funding_type`, `award_domain_name`, `is_global_south` | La convocatoria (award) en la OBT | Sí: se publican con la convocatoria |
| `work_domain_name`, `authors_count`, `cited_by_count` | Las publicaciones del investigador en la OBT | Sí, con el retraso con que OpenAlex las actualiza |
| `mismo_campo` | `award_field_name` y `work_field_name` | Sí: se calcula con `agregar_features` |
| `funder_en_pais_autor` | `funder_country_code` y `author_country_codes` | Sí: se calcula con `agregar_features` |

Todas vienen de la OBT, que se actualiza **una vez al día** con el pipeline de datos. Las variables que se conocen recién después del financiamiento (`funded_outputs_count`, `publication_year` del trabajo financiado) ya se quitaron en `pipeline_proyecto.py` y no entran en ningún patrón.

Esto refuerza la decisión: aunque el modelo respondiera en línea en milisegundos, sus datos de entrada solo cambian una vez por noche. Una predicción en línea a las 3 p. m. vería exactamente la misma información que el lote de la noche anterior.

## Qué se pierde con el patrón que descarto (en línea)

Al no servir en línea se pierde:

- **Los casos nuevos del día.** Un award publicado o un investigador que se registra hoy no tiene recomendaciones hasta el lote de la noche. Con convocatorias abiertas por semanas, ese día no cambia el resultado.
- **Preguntas que nadie previó**, como "¿y este award para este otro trabajo?". Para esas consultas sueltas está `servicio/api.py`: la misma preparación y el mismo modelo, detrás de una API.

Servir **todo** en línea costaría lo siguiente:

- Un servicio encendido todo el tiempo, con un límite de latencia por petición.
- Traer las features de Snowflake en cada consulta. Un `SELECT` a un warehouse tarda segundos, así que habría que montar una tienda de features de baja latencia aparte.
- Más pruebas y monitoreo de disponibilidad.

Todo eso compraría una frescura que los datos no tienen.

El lote también ahorra cómputo, porque el recorte por dominio evita puntuar pares imposibles. Su costo propio está anotado: predicciones que nadie llega a mirar (investigadores que no abren su lista), que el artículo de la lectura señala como la desventaja típica del lote.
