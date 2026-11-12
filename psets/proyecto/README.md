# GrantMatch (Subvenciones con Ruta)

GrantMatch recomienda convocatorias de financiamiento (awards) a investigadores. Para cada par (award, investigador), el modelo estima la probabilidad de que ese award financie un trabajo como el del investigador, usando los datos abiertos de OpenAlex: la convocatoria (financiador, país, tipo de ayuda, monto, duración, área) y el perfil del investigador (el área de sus publicaciones, sus citas, los países de sus coautores). Las recomendaciones se calculan por lotes cada noche ([`docs/despliegue.md`](docs/despliegue.md)) y se ordenan por esa probabilidad.

**Métrica de éxito:** AUC en prueba (si el modelo ordena un par real por encima de uno al azar), y para decidir qué se recomienda, el costo esperado con un falso negativo 10 veces más caro que un falso positivo: perder una convocatoria pertinente cuesta mucho más que leer una que no sirve. El umbral de decisión es el de menor costo en la validación (`modelado/validacion_proyecto.ipynb`).

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| OpenAlex, snapshot en S3 (`s3://openalex/data/parquet/`) | `topics` (~4.5 mil), `funders` (~46 mil) y `awards` (~17 M) completos; `works`: los primeros 2 archivos de la partición `updated_date` más reciente (~500 mil publicaciones) | `COPY INTO` de Snowflake desde el bucket público, sin credenciales ([`ingesta/`](ingesta/)) | Snowflake, `PSET2_DB.BRONZE` |
| La One Big Table del pipeline | Un par real (award, publicación financiada) por fila, con los atributos de los dos lados; muestra de 50 000 pares | [`modelado/exportar_obt.sql`](modelado/exportar_obt.sql) en Snowsight, descargado como CSV | `data/processed/obt_award_work_muestra.csv` |
| Pares con negativos | Los pares reales más, por cada uno, un par al azar con la publicación de otra fila (label 0), con las dos features | [`modelado/features_proyecto.ipynb`](modelado/features_proyecto.ipynb) | `data/processed/pares_proyecto.csv` |

Los datos no están en el repositorio. Para regenerarlos desde cero:

1. En Snowsight, con el rol ACCOUNTADMIN, corre [`ingesta/pipeline_snowsight.sql`](ingesta/pipeline_snowsight.sql) (Run All, ~20 min): OpenAlex → BRONZE → SILVER → GOLD → `OBT.OBT_AWARD_WORK`.
2. Corre [`modelado/exportar_obt.sql`](modelado/exportar_obt.sql), descarga el resultado y guárdalo como `data/processed/obt_award_work_muestra.csv`.
3. Corre `modelado/features_proyecto.ipynb`, que escribe `data/processed/pares_proyecto.csv`.

## Cómo correr el pipeline de datos

El pipeline completo es el del PSet 2 del equipo: Kestra (ingesta y orquestación) → dbt (SILVER y GOLD en Snowflake) → Spark (la OBT). Todo se levanta desde [`infra/`](infra/):

```
cd infra
cp .env.example .env              # credenciales de Snowflake y usuario de Kestra (no se sube)
# una vez: pega infra/snowflake_setup.sql en un worksheet de Snowflake y córrelo con ACCOUNTADMIN
docker compose up -d              # Kestra en localhost:8080, dbt y Spark
```

En Kestra, el flow `ingest_openalex_bronze` ([`orquestacion/flows/`](orquestacion/flows/)) carga BRONZE con `COPY INTO`, corre `dbt build` ([`warehouse/dbt/`](warehouse/dbt/)) y después el job de Spark ([`procesamiento/`](procesamiento/)), que escribe `OBT.OBT_AWARD_WORK`. Las decisiones de limpieza están en [`calidad/`](calidad/). Si la cuenta de Snowflake bloquea el login con contraseña (MFA), [`ingesta/pipeline_snowsight.sql`](ingesta/pipeline_snowsight.sql) hace lo mismo desde Snowsight.

## Cómo entrenar y evaluar el modelo

Todo corre desde [`modelado/`](modelado/), con el servidor de MLflow levantado (puerto 5050). [`modelado/pipeline_proyecto.py`](modelado/pipeline_proyecto.py) lee `data/processed/pares_proyecto.csv`, calcula las features y divide **por año de inicio del award**: entrenamiento con los awards que empezaron antes del año de corte (el percentil 80 de `award_start_year`) y prueba con los demás, porque el modelo tiene que recomendar convocatorias futuras. Un award y su par negativo caen siempre del mismo lado. Los notebooks importan ese archivo, no copian su código.

| Orden | Notebook | Qué hace | Qué deja |
|---|---|---|---|
| 1 | `explorar_proyecto.ipynb` | Resumen, objetivo y tres hallazgos | |
| 2 | `features_proyecto.ipynb` | Pares negativos y las dos features (`mismo_campo`, `funder_en_pais_autor`) | `data/processed/pares_proyecto.csv` |
| 3 | `python pipeline_proyecto.py` | Comprueba la división y la preparación | |
| 4 | `modelos_proyecto.ipynb` | Línea base, regresión logística y ensamblajes | `resultados_proyecto.csv` |
| 5 | `comparar_proyecto.ipynb` | El mejor ensamblaje contra una red neuronal y un LLM | `comparacion_proyecto.csv` |
| 6 | `experimentos_proyecto.ipynb` | Una corrida de MLflow por modelo (experimento `grantmatch`) y la final marcada con `candidato` = `final` | `corridas_proyecto.csv` |
| 7 | `validacion_proyecto.ipynb` | Validación cruzada por award y al azar, y la métrica de costo | `validacion_proyecto.csv` |

Las pruebas del pipeline y del servicio corren con `pytest psets/proyecto -v` desde la raíz del fork, y en GitHub Actions en cada push.

## Modelo final

⟨modelo⟩, corrida `⟨run_id⟩` del experimento `grantmatch` en MLflow: AUC en prueba ⟨X⟩ (línea base 0.5) y AUC ⟨X⟩ ± ⟨X⟩ en validación cruzada por award. Con el umbral de menor costo (⟨X⟩) recomienda ⟨X⟩ % de los awards pertinentes. El razonamiento completo está en [`docs/decision_modelo.md`](docs/decision_modelo.md).

## Cómo correr el servicio

El modelo se sirve por lotes cada noche (la razón está en [`docs/despliegue.md`](docs/despliegue.md)); `servicio/` es la API con que otro sistema le pide una predicción suelta al mismo modelo. Todo se corre desde `psets/proyecto/`.

**1. Exportar el modelo final desde MLflow.** Con el servidor de MLflow levantado (puerto 5050), el script busca en el experimento `grantmatch` la corrida marcada con `candidato` = `final` y la guarda como `servicio/modelo.joblib`, junto con `servicio/modelo_info.json` (su `run_id`). Ninguno de los dos se sube al repositorio.

```
python servicio/exportar_modelo.py            # la corrida final
python servicio/exportar_modelo.py <run_id>   # otra corrida, por ejemplo para volver a la anterior
```

**2. Probarlo sin contenedor.**

```
uvicorn servicio.api:app --port 8000
```

La documentación interactiva queda en el puerto 8000, ruta `/docs`. Las pruebas (`pytest psets/proyecto -v`, desde la raíz del fork) no necesitan el modelo exportado.

**3. Construir y levantar el contenedor.** La imagen se construye desde `psets/proyecto/` porque necesita `modelado/pipeline_proyecto.py` además de `servicio/`:

```
docker build -f servicio/Dockerfile -t grantmatch-servicio .
docker run -p 8000:8000 grantmatch-servicio
```

**4. Pedir una predicción.** Se manda el award y el perfil del investigador como se conocen. El servicio calcula `mismo_campo` y `funder_en_pais_autor` con `agregar_features` de `modelado/pipeline_proyecto.py`:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"award_field_name": "Physics and Astronomy", "funder_country_code": "US", "amount_usd": 250000, "duration_days": 1095, "funding_type": "grant", "award_domain_name": "Physical Sciences", "is_global_south": false, "work_field_name": "Physics and Astronomy", "author_country_codes": ["US", "DE"], "work_domain_name": "Physical Sciences", "authors_count": 4, "cited_by_count": 12}'
```

Respuesta (la probabilidad depende del modelo exportado):

```
{"probabilidad_financiamiento": ⟨X⟩, "recomendar": ⟨true/false⟩, "version_modelo": "⟨run_id⟩"}
```

`recomendar` es `true` cuando la probabilidad alcanza el umbral de menor costo de la validación (variable de entorno `UMBRAL`, 0.10 por defecto). Un par sin `award_field_name`, `funder_country_code`, `work_field_name` o `author_country_codes` se rechaza con 422: sin ellos las dos features saldrían 0 sin dar error.
