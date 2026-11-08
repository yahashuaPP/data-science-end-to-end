# Modelado de GrantMatch

GrantMatch recomienda convocatorias de financiamiento (awards) a investigadores. El modelo estima la probabilidad de que un award financie una publicación, a partir de los datos de OpenAlex que deja el pipeline del equipo en Snowflake (`OBT.OBT_AWARD_WORK`). Como la OBT solo guarda financiamientos reales, `features_proyecto.ipynb` arma pares negativos (award + publicación que no financió) y el problema queda como clasificación balanceada, medida con AUC.

## Orden para correrlo

Todo se corre desde esta carpeta (`psets/proyecto/modelado/`), con el kernel de Python 3.11 del Dev Container y el servidor de MLflow levantado (puerto 5050). Antes de cada commit: **Restart** y **Run All** en cada notebook.

1. **Datos.** En Snowsight, corre [`exportar_obt.sql`](./exportar_obt.sql) y guarda el resultado como `../data/raw/obt_award_work_muestra.csv`. `data/` no se sube (está en `.gitignore`).
2. **[`explorar_proyecto.ipynb`](./explorar_proyecto.ipynb)** (Semana 8): resumen, objetivo y tres hallazgos.
3. **[`features_proyecto.ipynb`](./features_proyecto.ipynb)** (Semana 9.1): arma los pares negativos, escribe `../data/processed/pares_proyecto.csv` y prueba las dos features (`mismo_campo`, `funder_en_pais_autor`).
4. **[`pipeline_proyecto.py`](./pipeline_proyecto.py)** (Semana 9.2): las features, la división por año del award y la preparación. Compruébalo con `python3 pipeline_proyecto.py`.
5. **[`modelos_proyecto.ipynb`](./modelos_proyecto.ipynb)** (Semana 10.1): línea base, regresión logística y ensamblajes → `resultados_proyecto.csv`.
6. **[`comparar_proyecto.ipynb`](./comparar_proyecto.ipynb)** (Semana 10.2): el mejor ensamblaje contra una red neuronal y un LLM → `comparacion_proyecto.csv`. A mitad del notebook hay que pasar `prompt_llm_proyecto.txt` a un LLM y guardar sus 10 respuestas en `respuestas_llm_proyecto.txt`.
7. **[`experimentos_proyecto.ipynb`](./experimentos_proyecto.ipynb)** (Semana 11.1): una corrida de MLflow por modelo en el experimento `grantmatch`, marca la final (`candidato` = `final`) y exporta `corridas_proyecto.csv`.
8. **[`validacion_proyecto.ipynb`](./validacion_proyecto.ipynb)** (Semana 11.2): validación cruzada por award y al azar del modelo final y el segundo, y la métrica de costo (falso negativo 10, falso positivo 1) → `validacion_proyecto.csv`.
