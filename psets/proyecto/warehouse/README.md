# Warehouse de GrantMatch

El warehouse es Snowflake (`PSET2_DB`), no DuckDB: BRONZE (crudo, lo carga la ingesta), SILVER (limpieza) y GOLD (star schema), construidos con el proyecto de dbt de [`dbt/`](./dbt/). El esquema se crea una vez con [`../infra/snowflake_setup.sql`](../infra/snowflake_setup.sql).

```
cd ../infra && docker compose up -d dbt
docker compose exec dbt dbt deps
docker compose exec dbt dbt build      # SILVER y GOLD con sus pruebas
```

La One Big Table que usa el modelado (`OBT.OBT_AWARD_WORK`) la construye el job de Spark de [`../procesamiento/`](../procesamiento/).
