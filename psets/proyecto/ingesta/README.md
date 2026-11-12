# Ingesta de GrantMatch

Los datos salen del snapshot público de OpenAlex en S3 (`s3://openalex/data/parquet/`): `topics`, `funders`, `awards` (completos) y una muestra de `works` (los primeros archivos de la partición `updated_date` más reciente). Snowflake los lee directo del bucket con `COPY INTO`, sin credenciales de AWS, y los deja en `PSET2_DB.BRONZE`.

Hay dos formas de correrla:

- **Con Kestra** (la del PSet 2): el flow [`../orquestacion/flows/01_ingest_openalex_bronze.yml`](../orquestacion/flows/01_ingest_openalex_bronze.yml) carga BRONZE con watermark y después corre dbt y Spark. Se levanta desde [`../infra/`](../infra/) con `docker compose up -d` (necesita `infra/.env` con las credenciales de Snowflake).
- **Solo con Snowsight**: [`pipeline_snowsight.sql`](./pipeline_snowsight.sql) hace la misma ingesta, los modelos de dbt compilados a SQL y la OBT en un solo worksheet (Run All, ~20 min). Sirve en una cuenta de Snowflake nueva, donde el login con contraseña de dbt y Kestra puede quedar bloqueado por MFA.
