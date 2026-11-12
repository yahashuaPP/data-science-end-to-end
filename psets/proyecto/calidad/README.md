# Calidad de los datos de GrantMatch

En GrantMatch la limpieza no es un script aparte: vive en la capa SILVER del proyecto de dbt, y el diagnóstico es una consulta de perfilado.

| Pieza | Dónde |
|---|---|
| Diagnóstico (faltantes, duplicados, valores inválidos, con ids A01-A15, F01-F03, T01-T02, W01-W09) | [`../warehouse/dbt/analyses/data_quality_profile.sql`](../warehouse/dbt/analyses/data_quality_profile.sql) |
| Limpieza (cada decisión documentada en el encabezado del modelo) | [`../warehouse/dbt/models/silver/`](../warehouse/dbt/models/silver/) |
| Pruebas de calidad (`not_null`, `unique`, `relationships` y reglas de negocio) | `schema.yml` de `silver/` y `gold/`, y [`../warehouse/dbt/tests/`](../warehouse/dbt/tests/) |

El perfilado se compila con `dbt compile --select data_quality_profile` (desde el contenedor `pset2-dbt`) y se corre en Snowsight.
