# Decisión del modelo de GrantMatch

> Para quien tiene que decidir si este modelo pasa a producción sin haber estado en clase. Cada número sale del registro de MLflow (experimento `grantmatch`, exportado en [`modelado/corridas_proyecto.csv`](../modelado/corridas_proyecto.csv)) o de la validación ([`modelado/validacion_proyecto.ipynb`](../modelado/validacion_proyecto.ipynb)).

**Qué predice el modelo.** Para un par (award, publicación), la probabilidad de que ese award haya financiado esa publicación. La publicación representa el perfil del investigador. GrantMatch usa esa probabilidad para ordenar las convocatorias que le recomienda a cada investigador. Los pares reales salen de la OBT del pipeline (OpenAlex). Los negativos se arman emparejando cada award con la publicación de otro par, así que las clases quedan balanceadas y una AUC de 0.5 equivale a no distinguir nada.

## 1. El modelo elegido

**⟨modelo⟩**, corrida `⟨run_id⟩` del experimento `grantmatch`, marcada con la etiqueta `candidato` = `final`. Es un `Pipeline` de scikit-learn 1.5.2 que incluye la preparación de [`modelado/pipeline_proyecto.py`](../modelado/pipeline_proyecto.py):
- mediana y escalado para las numéricas;
- moda y one-hot para las categóricas, que ignora las categorías nuevas;
- las tres binarias tal cual.

Se guardó como artefacto `modelo` de la corrida. Al cargarlo del registro, reproduce exactamente la AUC registrada (diferencia < 1e-9).

## 2. Contra qué lo comparé

Todos los candidatos usan la misma preparación, la misma división y la misma métrica:
- **División:** por año de inicio del award. Entrenamiento con los awards anteriores a ⟨año de corte⟩ y prueba con los demás.
- **Datos:** el mismo archivo, con huella `⟨huella⟩` (MD5 de `pares_proyecto.csv`).
- **Métrica:** AUC.

| Modelo | AUC entrenamiento | AUC prueba | Brecha | Segundos |
|---|---|---|---|---|
| Línea base (`DummyClassifier`, proporción de la clase) | 0.5 | 0.5 | 0 | ⟨X⟩ |
| Regresión logística C=1 | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |
| Regresión logística C=0.01 | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |
| Bosque aleatorio sin límite | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |
| Bosque aleatorio `max_depth=8` | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |
| Gradient boosting | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |
| Red neuronal (64 neuronas) | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ | ⟨X⟩ |

**La línea base.** Las clases están balanceadas por construcción, y un modelo que siempre responde la misma probabilidad no ordena nada (AUC 0.5). Cualquier candidato tiene que superarla para que sus features sirvan.

**Sobreajuste.** El bosque sin límite tiene la brecha más grande, ⟨X⟩. Limitar la profundidad a 8 la baja a ⟨X⟩, y la AUC de prueba ⟨sube / se mantiene⟩. En la logística, más regularización (C=0.01) ⟨cambia / casi no cambia⟩ la AUC de prueba: ⟨interpretación⟩.

**El LLM.** En 10 pares de prueba con 20 ejemplos, el LLM acertó ⟨X⟩ de 10, contra ⟨X⟩ del ensamblaje. Evaluar 10 000 pares costaría unos ⟨X⟩ dólares y no da una probabilidad para ordenar recomendaciones, así que no es un candidato para producción.

## 3. ¿La diferencia con el segundo es real?

Validación cruzada de 5 pliegues **por award** (`GroupKFold`): cada award cae entero en un solo pliegue, como una convocatoria nueva en producción. Al azar (`StratifiedKFold`), el mismo award puede quedar a la vez en entrenamiento y en validación, y el modelo se evalúa con convocatorias que ya vio. Esa versión está como comparación.

| Modelo | AUC por award (promedio ± desviación) | AUC al azar (promedio ± desviación) |
|---|---|---|
| ⟨final⟩ | ⟨X⟩ ± ⟨X⟩ | ⟨X⟩ ± ⟨X⟩ |
| ⟨segundo⟩ | ⟨X⟩ ± ⟨X⟩ | ⟨X⟩ ± ⟨X⟩ |

La AUC de cada pliegue está en `validacion_proyecto.csv`. La diferencia entre los dos es ⟨X⟩ y la desviación entre pliegues llega a ⟨X⟩. ⟨La diferencia supera la variación entre pliegues, así que la ventaja es real.⟩ / ⟨La diferencia queda dentro de la variación entre pliegues, así que los dos son equivalentes con estos datos y elijo el más simple y rápido.⟩ Al azar la AUC sale ⟨X⟩ más alta que por award: esa es la parte que el modelo "sabe" de awards que ya vio, y no estaría disponible con una convocatoria nueva.

## 4. El error más caro y la métrica del negocio

El error caro es el **falso negativo**: no recomendarle a un investigador un award que sí era para él. Puede costarle un financiamiento de decenas de miles de dólares; en esta muestra, la mediana de `amount_usd` es ⟨X⟩ dólares. Un **falso positivo** le cuesta unos minutos: leer la convocatoria y descartarla. Lo aproximo como **10 a 1**.

Con ese costo, el umbral de menor costo esperado es **⟨X⟩**, no 0.5. Con ese umbral:
- se recomienda el ⟨X⟩ % de los awards pertinentes (recall);
- se pierden ⟨X⟩ y se recomiendan ⟨X⟩ de más;
- el costo es ⟨X⟩, contra ⟨X⟩ con el umbral de 0.5.

El servicio usa ese umbral (variable `UMBRAL` de [`servicio/api.py`](../servicio/api.py)).

## 5. Costo y explicabilidad

El modelo final se entrena en **⟨X⟩ segundos** con ⟨X⟩ pares, y puntúa el lote nocturno en segundos. Las columnas que más pesan son ⟨las 3 primeras de la importancia⟩, así que alguien del equipo puede explicar una recomendación con razones como "el award es de tu misma área" o "el financiador es de tu país".

⟨Si el final es un ensamblaje: la importancia dice qué columnas usa en general, no por qué recomienda un par en particular. La logística lo diría con sus coeficientes, a cambio de ⟨X⟩ de AUC.⟩ La red neuronal no ofrece ninguna de las dos explicaciones, y ⟨no gana / gana solo ⟨X⟩⟩ en AUC.

## 6. Cómo obtener exactamente el mismo número

Hace falta:

1. **Los mismos datos.** El mismo `pares_proyecto.csv`, comprobado con su huella `⟨huella⟩`, que está registrada en cada corrida. Para regenerarlo: la muestra de la OBT ([`modelado/exportar_obt.sql`](../modelado/exportar_obt.sql), con `SAMPLE`) y los negativos de `features_proyecto.ipynb` con semilla 42. Ojo: `SAMPLE` en Snowflake no tiene semilla, así que una exportación nueva da otra muestra y otra huella. El CSV exportado es el que hay que conservar.
2. **La misma división.** El año de corte ⟨X⟩, registrado como parámetro `anio_de_corte`.
3. **El mismo código y las mismas versiones.** `pipeline_proyecto.py` en el commit del tag `pset-03`, scikit-learn 1.5.2 y Python 3.11 (los del `requirements.txt` del curso y del Dev Container).
4. **Los mismos hiperparámetros**, con `random_state=42`. Están todos registrados en la corrida.

**Lo que quedó registrado en MLflow:**
- hiperparámetros;
- división y año de corte;
- huella de los datos y filas de entrenamiento;
- AUC de entrenamiento y de prueba;
- segundos de entrenamiento;
- el pipeline entrenado como artefacto.

**Lo que no:** el CSV mismo (solo su huella) y el commit del código. Para cerrarlo habría que registrar el commit como etiqueta y subir el CSV como artefacto, o versionarlo con DVC.

## Decisión

⟨Paso a producción el ⟨modelo⟩ con umbral ⟨X⟩ / Recomiendo ⟨…⟩⟩, por la AUC por award (⟨X⟩ ± ⟨X⟩ contra 0.5 de la línea base), el costo con el umbral elegido y ⟨la explicabilidad / el tiempo de entrenamiento⟩.

Límites conocidos:
- Los works son una muestra de una sola partición de OpenAlex.
- Los negativos son al azar, más fáciles que los "casi pertinentes" que verá en producción, así que la AUC real será menor.
- Hay que reentrenar cuando OpenAlex publique un snapshot nuevo.
