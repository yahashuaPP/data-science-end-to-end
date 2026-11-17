# Decisión del modelo de GrantMatch

> Para quien tiene que decidir si este modelo pasa a producción sin haber estado en clase. Cada número sale del registro de MLflow (experimento `grantmatch`, exportado en [`modelado/corridas_proyecto.csv`](../modelado/corridas_proyecto.csv)) o de la validación ([`modelado/validacion_proyecto.ipynb`](../modelado/validacion_proyecto.ipynb)).

**Qué predice el modelo.** Para un par (award, publicación), la probabilidad de que ese award haya financiado esa publicación. La publicación representa el perfil del investigador. GrantMatch usa esa probabilidad para ordenar las convocatorias que le recomienda a cada investigador.

**De dónde salen los pares.** Los pares reales salen de una muestra de 50 000 filas de la OBT del pipeline (OpenAlex). Los negativos se arman emparejando cada award con la publicación de otro par, así que las clases quedan balanceadas y una AUC de 0.5 equivale a no distinguir nada. El modelo usa los 16 440 pares cuyo award tiene año de inicio: solo el 16.4 % de los awards lo trae, y sin año no se puede dividir por fecha.

## 1. El modelo elegido

**Gradient boosting** (`GradientBoostingClassifier`, 100 árboles de profundidad 3, `random_state=42`), corrida `cabc6e59ca034fa897c7070f7d58d16b` del experimento `grantmatch`, marcada con la etiqueta `candidato` = `final`.

Es un `Pipeline` de scikit-learn 1.5.2 que incluye la preparación de [`modelado/pipeline_proyecto.py`](../modelado/pipeline_proyecto.py):
- mediana y escalado para las numéricas;
- moda y one-hot para las categóricas, que ignora las categorías nuevas;
- las dos binarias (`mismo_campo` y `funder_en_pais_autor`) tal cual.

Se guardó como artefacto `modelo` de la corrida. Al cargarlo del registro, reproduce exactamente la AUC registrada.

## 2. Contra qué lo comparé

Todos los candidatos usan la misma preparación, la misma división y la misma métrica:
- **División:** por año de inicio del award. Entrenamiento con los awards anteriores a 2020 (11 884 pares) y prueba con los de 2020 en adelante (4 556 pares). Un award y su par negativo caen siempre del mismo lado.
- **Datos:** el mismo archivo, con huella `acdaa7713ac3` (MD5 de `pares_proyecto.csv`).
- **Métrica:** AUC.

| Modelo | AUC entrenamiento | AUC prueba | Brecha | Segundos |
|---|---|---|---|---|
| Línea base (`DummyClassifier`, proporción de la clase) | 0.500 | 0.500 | 0 | 0.03 |
| Regresión logística C=1 | 0.871 | 0.881 | −0.010 | 0.06 |
| Regresión logística C=0.01 | 0.867 | 0.879 | −0.012 | 0.04 |
| Bosque aleatorio sin límite | 0.999 | 0.883 | 0.116 | 1.05 |
| Bosque aleatorio `max_depth=8` | 0.908 | 0.895 | 0.013 | 0.52 |
| **Gradient boosting** | **0.902** | **0.902** | **−0.000** | **1.05** |
| Red neuronal (64 neuronas, 2 177 parámetros) | 0.894 | 0.892 | 0.003 | 3.65 |

**La línea base.** Las clases están balanceadas por construcción, y un modelo que siempre responde la misma probabilidad no ordena nada (AUC 0.5). Cualquier candidato tiene que superarla para que sus features sirvan. Todos la superan por casi 0.4.

**Sobreajuste.** El bosque sin límite memoriza: 0.999 en entrenamiento contra 0.883 en prueba. Limitar la profundidad a 8 baja la brecha a 0.013 y sube la AUC de prueba a 0.895. En la logística, más regularización (C=0.01) casi no cambia la AUC de prueba (0.881 contra 0.879): no estaba sobreajustada. El gradient boosting tiene la misma AUC en entrenamiento y en prueba.

**El LLM.** Pasé el mismo prompt (10 pares de prueba, 20 de entrenamiento como ejemplos) a cuatro LLM, cada uno en una conversación nueva. Acertaron 10 de 10 (Claude Sonnet 5.5), 8 (Gemini), 6 (GPT) y 3 (Claude Haiku 5.5), contra 9 del gradient boosting y 9 de la red. Con 10 pares el mejor LLM no se distingue del gradient boosting, y entre LLM el resultado va de 3 a 10 con el mismo prompt. Evaluar 10 000 pares costaría unos 3.4 dólares (a 4 dólares por millón de tokens de entrada) y no da una probabilidad para ordenar recomendaciones, así que no es un candidato para producción.

## 3. ¿La diferencia con el segundo es real?

Validación cruzada de 5 pliegues **por award** (`GroupKFold`): cada award cae entero en un solo pliegue, como una convocatoria nueva en producción. Al azar (`KFold` con `shuffle`), el mismo award puede quedar a la vez en entrenamiento y en validación. Esa versión está como comparación.

| Modelo | AUC por award (promedio ± desviación) | AUC al azar (promedio ± desviación) |
|---|---|---|
| Gradient boosting | 0.8972 ± 0.0048 | 0.8968 ± 0.0042 |
| Bosque `max_depth=8` | 0.8934 ± 0.0043 | 0.8938 ± 0.0033 |

La AUC de cada pliegue está en `validacion_proyecto.csv`. El peor pliegue del gradient boosting da 0.889.

**La diferencia entre los dos (0.0038) es menor que la desviación entre pliegues (0.0048): con estos datos no se distinguen.** Entre dos empatados me quedo con el gradient boosting porque es el más simple de servir:
- su modelo guardado pesa 172 KB contra 3.8 MB del bosque (100 árboles de profundidad 3 contra 200 de profundidad 8), y predice más rápido;
- la diferencia en entrenamiento (1.05 s contra 0.52 s) no importa en un lote nocturno.

Al azar y por award dan casi lo mismo, por dos razones: casi cada award aparece una sola vez (7 795 awards en 8 220 pares reales), y las features que más pesan comparan los dos lados del par, así que no dependen de qué award es.

## 4. El error más caro y la métrica del negocio

El error caro es el **falso negativo**: no recomendarle a un investigador un award que sí era para él. Puede costarle un financiamiento grande: en esta muestra, la mediana de `amount_usd` es 562 058 dólares. Un **falso positivo** le cuesta unos minutos: leer la convocatoria y descartarla. Lo aproximo como **10 a 1**.

Con ese costo, medido con predicciones de la validación cruzada por award, el umbral de menor costo es **0.10**, no 0.5:

| Umbral | Recall | Pertinentes perdidos | Recomendaciones de más | Costo |
|---|---|---|---|---|
| 0.05 | 0.999 | 7 | 8 127 | 8 197 |
| **0.10** | **0.982** | **144** | **5 891** | **7 331** |
| 0.50 | 0.853 | 1 209 | 1 429 | 13 519 |

Con 0.10 se recomienda el 98.2 % de los awards pertinentes y el costo baja 46 % respecto a 0.5. El servicio usa ese umbral (variable `UMBRAL` de [`servicio/api.py`](../servicio/api.py)).

## 5. Costo y explicabilidad

El modelo final se entrena en **1.05 segundos** con 11 884 pares, y puntúa el lote nocturno en segundos.

Las columnas que más pesan en el gradient boosting son:
- `funder_en_pais_autor` (0.79 de la importancia);
- `amount_usd` (0.06);
- `authors_count` (0.05);
- `mismo_campo` (0.03).

Así, alguien del equipo puede explicar una recomendación con razones como "el financiador es de tu país" o "el award es de tu misma área". La importancia dice qué columnas usa en general, no por qué recomienda un par en particular. La logística lo diría con sus coeficientes, a cambio de 0.021 de AUC en prueba. La red neuronal no ofrece ninguna de las dos explicaciones y no gana en AUC (0.892).

## 6. Cómo obtener exactamente el mismo número

Hace falta:

1. **Los mismos datos.** El mismo `pares_proyecto.csv`, comprobado con su huella `acdaa7713ac3`, que está registrada en cada corrida. Para regenerarlo:
   - la muestra de la OBT ([`modelado/exportar_obt.sql`](../modelado/exportar_obt.sql), con `SAMPLE`);
   - los negativos de `features_proyecto.ipynb`, con semilla 42.

   Ojo: `SAMPLE` en Snowflake no tiene semilla, así que una exportación nueva da otra muestra y otra huella. El CSV exportado es el que hay que conservar.
2. **La misma división.** El año de corte 2020, registrado como parámetro `anio_de_corte`.
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

Recomiendo pasar a producción el **gradient boosting** (`cabc6e59ca034fa897c7070f7d58d16b`) con umbral **0.10**:
- AUC por award de 0.897 ± 0.005 contra 0.5 de la línea base;
- empata con el bosque y es el más liviano de servir;
- con el umbral elegido recomienda el 98 % de los awards pertinentes al menor costo.

Límites conocidos:
- **Sesgo de la muestra.** Solo se usan los awards con año de inicio (16 % de la OBT, sobre todo de agencias grandes: NSF, NSFC, JSPS, Comisión Europea), así que no sé si funciona igual con los demás.
- **Negativos fáciles.** Los negativos son al azar, más fáciles que los "casi pertinentes" que verá en producción: la AUC real será menor.
- **Una feature domina.** Casi toda la señal viene de `funder_en_pais_autor`.
- **Muestra pequeña de works.** Los works son de una sola partición de OpenAlex, y hay que reentrenar cuando salga un snapshot nuevo.
