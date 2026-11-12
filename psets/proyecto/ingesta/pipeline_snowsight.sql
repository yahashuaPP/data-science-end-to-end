-- ============================================================================
-- GrantMatch completo en Snowsight: OpenAlex (S3) -> BRONZE -> SILVER -> GOLD -> OBT
--
-- Es el pipeline del PSet 2 (pset_2/) sin Docker: la ingesta de
-- kestra/flows/01_ingest_openalex_bronze.yml, los 18 modelos de dbt/models/
-- compilados a SQL (primera construccion completa) y la OBT de
-- spark/jobs/build_obt.py escrita en SQL. Mismas tablas, mismas columnas.
--
-- Como correrlo:
--   1. Snowsight > Projects > Worksheets > + SQL Worksheet. Pega todo este archivo.
--   2. Arriba a la derecha: rol ACCOUNTADMIN.
--   3. Run All (Ctrl+Shift+Enter). Tarda ~15-25 min con un warehouse XSMALL
--      (lo mas lento: 17 M de awards y el abstract de ~500 mil works).
--   4. Al final corre exportar_obt.sql (psets/proyecto/modelado/) y descarga el CSV.
--
-- Si algo falla a la mitad, corrige y vuelve a correr desde ese paso: cada
-- paso es CREATE OR REPLACE (o COPY INTO, que no recarga archivos ya cargados).
-- ============================================================================

-- ---------------------------------------------------------------- 0. Setup
-- (docs/snowflake_setup.sql, sin el rol de equipo: aqui se trabaja con ACCOUNTADMIN)
USE ROLE ACCOUNTADMIN;

CREATE WAREHOUSE IF NOT EXISTS PSET2_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;
USE WAREHOUSE PSET2_WH;

CREATE DATABASE IF NOT EXISTS PSET2_DB;
CREATE SCHEMA IF NOT EXISTS PSET2_DB.BRONZE;
CREATE SCHEMA IF NOT EXISTS PSET2_DB.SILVER;
CREATE SCHEMA IF NOT EXISTS PSET2_DB.GOLD;
CREATE SCHEMA IF NOT EXISTS PSET2_DB.OBT;
USE DATABASE PSET2_DB;

-- ---------------------------------------------------------------- 1. BRONZE
-- Stage sobre el bucket publico de OpenAlex (no necesita credenciales de AWS).
CREATE OR REPLACE STAGE PSET2_DB.BRONZE.OPENALEX_STAGE
  URL = 's3://openalex/data/parquet/'
  FILE_FORMAT = (TYPE = PARQUET);

CREATE TABLE IF NOT EXISTS PSET2_DB.BRONZE.RAW_OPENALEX_TOPICS (
  RAW VARIANT NOT NULL, _SOURCE_FILE VARCHAR NOT NULL, _SOURCE_ROW_NUMBER NUMBER(38, 0), _LOADED_AT TIMESTAMP_LTZ NOT NULL);
CREATE TABLE IF NOT EXISTS PSET2_DB.BRONZE.RAW_OPENALEX_FUNDERS (
  RAW VARIANT NOT NULL, _SOURCE_FILE VARCHAR NOT NULL, _SOURCE_ROW_NUMBER NUMBER(38, 0), _LOADED_AT TIMESTAMP_LTZ NOT NULL);
CREATE TABLE IF NOT EXISTS PSET2_DB.BRONZE.RAW_OPENALEX_AWARDS (
  RAW VARIANT NOT NULL, _SOURCE_FILE VARCHAR NOT NULL, _SOURCE_ROW_NUMBER NUMBER(38, 0), _LOADED_AT TIMESTAMP_LTZ NOT NULL);
CREATE TABLE IF NOT EXISTS PSET2_DB.BRONZE.RAW_OPENALEX_WORKS (
  RAW VARIANT NOT NULL, _SOURCE_FILE VARCHAR NOT NULL, _SOURCE_ROW_NUMBER NUMBER(38, 0), _LOADED_AT TIMESTAMP_LTZ NOT NULL);

-- Catalogos completos: topics (~4.5 mil), funders (~46 mil), awards (~17 M, ~2 min).
COPY INTO PSET2_DB.BRONZE.RAW_OPENALEX_TOPICS (RAW, _SOURCE_FILE, _SOURCE_ROW_NUMBER, _LOADED_AT)
  FROM (SELECT $1, METADATA$FILENAME, METADATA$FILE_ROW_NUMBER, CURRENT_TIMESTAMP()
        FROM @PSET2_DB.BRONZE.OPENALEX_STAGE/topics/)
  PATTERN = '.*[.]parquet' ON_ERROR = ABORT_STATEMENT;

COPY INTO PSET2_DB.BRONZE.RAW_OPENALEX_FUNDERS (RAW, _SOURCE_FILE, _SOURCE_ROW_NUMBER, _LOADED_AT)
  FROM (SELECT $1, METADATA$FILENAME, METADATA$FILE_ROW_NUMBER, CURRENT_TIMESTAMP()
        FROM @PSET2_DB.BRONZE.OPENALEX_STAGE/funders/)
  PATTERN = '.*[.]parquet' ON_ERROR = ABORT_STATEMENT;

COPY INTO PSET2_DB.BRONZE.RAW_OPENALEX_AWARDS (RAW, _SOURCE_FILE, _SOURCE_ROW_NUMBER, _LOADED_AT)
  FROM (SELECT $1, METADATA$FILENAME, METADATA$FILE_ROW_NUMBER, CURRENT_TIMESTAMP()
        FROM @PSET2_DB.BRONZE.OPENALEX_STAGE/awards/)
  PATTERN = '.*[.]parquet' ON_ERROR = ABORT_STATEMENT;

-- Works: como la primera corrida de Kestra, la particion updated_date mas reciente
-- (la del ultimo snapshot) y sus 2 primeros archivos. Con la del 23-sep-2026 eso
-- fueron 558 mil works y una OBT de 313 mil pares. Un dia completo pesa ~136 GB:
-- no subas works_archivos de 2-3 en la cuenta de prueba.
EXECUTE IMMEDIATE $$
DECLARE
  works_archivos INTEGER DEFAULT 2;
  ultima VARCHAR;
  copia VARCHAR;
BEGIN
  EXECUTE IMMEDIATE 'LIST @PSET2_DB.BRONZE.OPENALEX_STAGE/works/ PATTERN = ''.*part_0000[.]parquet''';
  SELECT MAX(REGEXP_SUBSTR("name", 'updated_date=([0-9]{4}-[0-9]{2}-[0-9]{2})', 1, 1, 'e', 1))
    INTO :ultima
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  IF (ultima IS NULL) THEN
    RETURN 'No encontre particiones de works en el stage: revisa el paso del stage';
  END IF;
  copia := 'COPY INTO PSET2_DB.BRONZE.RAW_OPENALEX_WORKS (RAW, _SOURCE_FILE, _SOURCE_ROW_NUMBER, _LOADED_AT) ' ||
           'FROM (SELECT $1, METADATA$FILENAME, METADATA$FILE_ROW_NUMBER, CURRENT_TIMESTAMP() ' ||
           'FROM @PSET2_DB.BRONZE.OPENALEX_STAGE/works/) ' ||
           'PATTERN = ''.*updated_date=' || ultima || '/part_000[0-' || (works_archivos - 1)::VARCHAR || '][.]parquet'' ' ||
           'ON_ERROR = ABORT_STATEMENT';
  EXECUTE IMMEDIATE copia;
  RETURN 'works cargados de la particion updated_date=' || ultima;
END;
$$;

-- Evidencia de la carga (anotala para la seccion de limitaciones).
SELECT 'TOPICS' AS tabla, COUNT(*) AS filas, COUNT(DISTINCT _SOURCE_FILE) AS archivos FROM PSET2_DB.BRONZE.RAW_OPENALEX_TOPICS
UNION ALL SELECT 'FUNDERS', COUNT(*), COUNT(DISTINCT _SOURCE_FILE) FROM PSET2_DB.BRONZE.RAW_OPENALEX_FUNDERS
UNION ALL SELECT 'AWARDS', COUNT(*), COUNT(DISTINCT _SOURCE_FILE) FROM PSET2_DB.BRONZE.RAW_OPENALEX_AWARDS
UNION ALL SELECT 'WORKS', COUNT(*), COUNT(DISTINCT _SOURCE_FILE) FROM PSET2_DB.BRONZE.RAW_OPENALEX_WORKS;

-- ---------------------------------------------------------------- 2. SILVER y 3. GOLD (dbt compilado)
-- SILVER · slv_topics  (models/silver/slv_topics.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_TOPICS AS
with source as (

    select raw, _loaded_at
    from PSET2_DB.BRONZE.RAW_OPENALEX_TOPICS

),

typed as (

    select
        nullif(split_part(trim(raw:id::varchar), '/', -1), '')                               as topic_id,
        nullif(trim(regexp_replace(regexp_replace(raw:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                as topic_name,
        nullif(trim(regexp_replace(regexp_replace(raw:description::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                 as topic_description,
        raw:keywords                                        as keywords,
        nullif(split_part(trim(raw:subfield:id::varchar), '/', -1), '')                      as subfield_id,
        nullif(trim(regexp_replace(regexp_replace(raw:subfield:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')       as subfield_name,
        nullif(split_part(trim(raw:field:id::varchar), '/', -1), '')                         as field_id,
        nullif(trim(regexp_replace(regexp_replace(raw:field:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')          as field_name,
        nullif(split_part(trim(raw:domain:id::varchar), '/', -1), '')                        as domain_id,
        nullif(trim(regexp_replace(regexp_replace(raw:domain:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')         as domain_name,
        try_cast(raw:works_count::varchar as integer)       as works_count,
        try_to_timestamp_ntz(raw:updated_date::varchar)     as updated_at,
        _loaded_at
    from source

)

select *
from typed
where topic_id is not null
qualify row_number() over (
    partition by topic_id
    order by updated_at desc nulls last, _loaded_at desc
) = 1;

-- SILVER · slv_funders  (models/silver/slv_funders.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_FUNDERS AS
with source as (

    select raw, _loaded_at
    from PSET2_DB.BRONZE.RAW_OPENALEX_FUNDERS

),

typed as (

    select
        nullif(split_part(trim(raw:id::varchar), '/', -1), '')                                   as funder_id,
        nullif(trim(regexp_replace(regexp_replace(raw:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                    as funder_name,
        nullif(trim(regexp_replace(regexp_replace(raw:description::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                     as funder_description,
        nullif(upper(trim(raw:country_code::varchar)), '')      as country_code,
        try_cast(raw:is_global_south::varchar as boolean)       as is_global_south,
        nullif(split_part(trim(raw:ids:ror::varchar), '/', -1), '')                              as ror_id,
        nullif(trim(raw:homepage_url::varchar), '')             as homepage_url,
        iff(try_cast(raw:awards_count::varchar as integer) >= 0,
            try_cast(raw:awards_count::varchar as integer), null) as awards_count,
        iff(try_cast(raw:works_count::varchar as integer) >= 0,
            try_cast(raw:works_count::varchar as integer), null)  as works_count,
        try_to_timestamp_ntz(raw:updated_date::varchar)         as updated_at,
        _loaded_at
    from source

)

select *
from typed
where funder_id is not null
qualify row_number() over (
    partition by funder_id
    order by updated_at desc nulls last, _loaded_at desc
) = 1;

-- SILVER · slv_awards_history  (models/silver/slv_awards_history.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_AWARDS_HISTORY AS
with source as (

    select raw, _source_file, _loaded_at
    from PSET2_DB.BRONZE.RAW_OPENALEX_AWARDS

),

typed as (

    select
        nullif(split_part(trim(raw:id::varchar), '/', -1), '')                                       as award_id,
        try_to_timestamp_ntz(raw:updated_date::varchar)             as updated_at,
        try_to_timestamp_ntz(raw:created_date::varchar)             as created_at,

        nullif(trim(regexp_replace(regexp_replace(raw:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                        as award_title,
        nullif(trim(regexp_replace(regexp_replace(raw:description::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                         as award_description,
        nullif(trim(raw:funder_award_id::varchar), '')              as funder_award_id,
        nullif(trim(raw:funder_scheme::varchar), '')                as funder_scheme,
        nullif(lower(trim(raw:funding_type::varchar)), '')          as funding_type,

        nullif(split_part(trim(raw:funder:id::varchar), '/', -1), '')                                as funder_id,
        nullif(trim(regexp_replace(regexp_replace(raw:funder:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                 as funder_name,

        try_cast(raw:amount::varchar as number(18, 2))              as amount_raw,
        nullif(upper(trim(raw:currency::varchar)), '')              as currency,

        try_to_date(raw:start_date::varchar)                        as start_date_raw,
        try_cast(raw:start_year::varchar as integer)                as start_year,
        -- Fecha de inicio final: la exacta o, si falta, el 1 de enero de start_year.
        coalesce(start_date_raw,
                 iff(start_year between 1900 and 2100, date_from_parts(start_year, 1, 1), null))
                                                                    as start_date,
        try_to_date(raw:end_date::varchar)                          as end_date_raw,
        try_cast(raw:end_year::varchar as integer)                  as end_year,

        nullif(split_part(trim(raw:primary_topic:id::varchar), '/', -1), '')                         as primary_topic_id,
        raw:institution_awarded                                     as institutions_awarded,
        nullif(split_part(trim(raw:institution_awarded[0]:id::varchar), '/', -1), '')                as lead_institution_id,
        coalesce(array_size(raw:institution_awarded), 0)            as institutions_count,

        raw:funded_outputs                                          as funded_outputs,
        coalesce(try_cast(raw:funded_outputs_count::varchar as integer),
                 array_size(raw:funded_outputs), 0)                 as funded_outputs_count,

        nullif(lower(trim(raw:doi::varchar)), '')                   as doi,
        nullif(trim(raw:landing_page_url::varchar), '')             as landing_page_url,
        nullif(trim(raw:provenance::varchar), '')                   as provenance,

        _source_file,
        _loaded_at

    from source

),

cleaned as (

    select
        award_id,
        updated_at,
        created_at,
        award_title,
        award_description,
        award_description is not null                               as has_description,
        funder_award_id,
        funder_scheme,
        funding_type,
        funder_id,
        funder_name,

        iff(amount_raw > 0, amount_raw, null)                       as amount,
        coalesce(amount_raw <= 0, false)                            as is_amount_invalid,
        currency,
        iff(currency = 'USD' and amount_raw > 0, amount_raw, null)  as amount_usd,

        start_date,
        coalesce(start_date_raw is null and start_year between 1900 and 2100, false)
                                                                    as is_start_date_from_year,
        start_year,
        -- Contra la fecha de inicio final (exacta o derivada del año), no solo la exacta.
        iff(end_date_raw < start_date, null, end_date_raw)          as end_date,
        coalesce(end_date_raw < start_date, false)                  as has_inconsistent_dates,
        end_year,

        primary_topic_id,
        institutions_awarded,
        lead_institution_id,
        institutions_count,
        funded_outputs,
        funded_outputs_count,
        doi,
        landing_page_url,
        provenance,
        _source_file,
        _loaded_at

    from typed
    where award_id is not null

),

deduplicated as (

    select *
    from cleaned
    qualify row_number() over (
        partition by award_id, updated_at
        order by _loaded_at desc, _source_file desc
    ) = 1

)

select d.*
from deduplicated d;

-- SILVER · slv_awards  (models/silver/slv_awards.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_AWARDS AS
with changed as (

    select *
    from PSET2_DB.SILVER.SLV_AWARDS_HISTORY

),

latest as (

    select *
    from changed
    qualify row_number() over (
        partition by award_id
        order by updated_at desc nulls last, _loaded_at desc
    ) = 1

)

select l.*
from latest l;

-- SILVER · slv_works_history  (models/silver/slv_works_history.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_WORKS_HISTORY AS
with source as (

    select raw, _source_file, _loaded_at
    from PSET2_DB.BRONZE.RAW_OPENALEX_WORKS

),

typed as (

    select
        nullif(split_part(trim(raw:id::varchar), '/', -1), '')                                       as work_id,
        try_to_timestamp_ntz(raw:updated_date::varchar)             as updated_at,
        nullif(lower(trim(raw:doi::varchar)), '')                   as doi,
        coalesce(nullif(trim(regexp_replace(regexp_replace(raw:title::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), ''),
                 nullif(trim(regexp_replace(regexp_replace(raw:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), ''))              as title,
        try_cast(raw:publication_year::varchar as integer)          as publication_year_raw,
        try_to_date(raw:publication_date::varchar)                  as publication_date,
        nullif(lower(trim(raw:type::varchar)), '')                  as work_type,
        nullif(lower(trim(raw:language::varchar)), '')              as language,
        try_cast(raw:cited_by_count::varchar as integer)            as cited_by_count_raw,
        try_cast(raw:fwci::varchar as float)                        as fwci,
        coalesce(try_cast(raw:is_retracted::varchar as boolean), false) as is_retracted,
        nullif(split_part(trim(raw:primary_topic:id::varchar), '/', -1), '')                         as primary_topic_id,
        raw:abstract_inverted_index                                 as abstract_inverted_index,
        raw:authorships                                             as authorships,
        coalesce(array_size(raw:authorships), 0)                    as authors_count,
        raw:awards                                                  as awards,
        coalesce(array_size(raw:awards), 0)                         as awards_count,
        _source_file,
        _loaded_at
    from source

),

cleaned as (

    select
        work_id,
        updated_at,
        doi,
        title,
        title is not null                                           as has_title,
        iff(publication_year_raw between 1500
                                     and year(current_date()) + 1,
            publication_year_raw, null)                             as publication_year,
        coalesce(not (publication_year_raw between 1500
                                               and year(current_date()) + 1), false)
                                                                    as is_publication_year_invalid,
        publication_date,
        work_type,
        language,
        iff(cited_by_count_raw >= 0, cited_by_count_raw, null)      as cited_by_count,
        fwci,
        is_retracted,
        primary_topic_id,
        abstract_inverted_index,
        authorships,
        authors_count,
        awards,
        awards_count,
        _source_file,
        _loaded_at
    from typed
    where work_id is not null

),

deduplicated as (

    select *
    from cleaned
    qualify row_number() over (
        partition by work_id, updated_at
        order by _loaded_at desc, _source_file desc
    ) = 1

)

select d.*
from deduplicated d;

-- SILVER · slv_works  (models/silver/slv_works.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_WORKS AS
with changed as (

    select *
    from PSET2_DB.SILVER.SLV_WORKS_HISTORY

),

latest as (

    select *
    from changed
    qualify row_number() over (
        partition by work_id
        order by updated_at desc nulls last, _loaded_at desc
    ) = 1

),

abstract_words as (

    select
        l.work_id,
        w.key                       as word,
        p.value::integer            as position
    from latest l,
        lateral flatten(input => iff(is_object(l.abstract_inverted_index),
                                     l.abstract_inverted_index,
                                     try_parse_json(l.abstract_inverted_index::varchar))) w,
        lateral flatten(input => w.value) p

),

abstracts as (

    select
        work_id,
        listagg(word, ' ') within group (order by position) as abstract_text
    from abstract_words
    group by work_id

),

final as (

    select
        l.work_id,
        l.updated_at,
        l.doi,
        l.title,
        l.has_title,
        a.abstract_text,
        a.abstract_text is not null      as has_abstract,
        l.publication_year,
        l.is_publication_year_invalid,
        l.publication_date,
        l.work_type,
        l.language,
        l.cited_by_count,
        l.fwci,
        l.is_retracted,
        l.primary_topic_id,
        l.authorships,
        l.authors_count,
        l.awards,
        l.awards_count,
        l._source_file,
        l._loaded_at
    from latest l
    left join abstracts a
        on a.work_id = l.work_id

)

select f.*
from final f;

-- SILVER · slv_work_authorships  (models/silver/slv_work_authorships.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_WORK_AUTHORSHIPS AS
with works as (

    select work_id, authorships, _loaded_at
    from PSET2_DB.SILVER.SLV_WORKS

),

flattened as (

    select
        w.work_id,
        nullif(split_part(trim(a.value:author:id::varchar), '/', -1), '')                       as author_id,
        nullif(trim(regexp_replace(regexp_replace(a.value:author:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')        as author_name,
        nullif(split_part(trim(a.value:author:orcid::varchar), '/', -1), '')                    as orcid,
        nullif(lower(trim(a.value:author_position::varchar)), '') as author_position,
        coalesce(try_cast(a.value:is_corresponding::varchar as boolean), false) as is_corresponding,
        nullif(split_part(trim(a.value:institutions[0]:id::varchar), '/', -1), '')              as institution_id,
        nullif(trim(regexp_replace(regexp_replace(a.value:institutions[0]:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '') as institution_name,
        nullif(upper(trim(a.value:institutions[0]:country_code::varchar)), '') as institution_country_code,
        a.index                                                as author_order,
        w._loaded_at
    from works w,
        lateral flatten(input => w.authorships) a

)

select *
from flattened
where author_id is not null
qualify row_number() over (partition by work_id, author_id order by author_order) = 1;

-- SILVER · slv_work_awards  (models/silver/slv_work_awards.sql)
CREATE OR REPLACE TABLE PSET2_DB.SILVER.SLV_WORK_AWARDS AS
with works as (

    select work_id, awards, _loaded_at
    from PSET2_DB.SILVER.SLV_WORKS

),

flattened as (

    select
        w.work_id,
        nullif(split_part(trim(a.value:id::varchar), '/', -1), '')                          as award_id,
        nullif(split_part(trim(a.value:funder_id::varchar), '/', -1), '')                   as funder_id,
        nullif(trim(a.value:funder_award_id::varchar), '') as funder_award_id,
        a.index                                            as award_order,
        w._loaded_at
    from works w,
        lateral flatten(input => w.awards) a

)

select *
from flattened
where award_id is not null
qualify row_number() over (partition by work_id, award_id order by award_order) = 1;

-- GOLD · dim_date  (models/gold/dim_date.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_DATE AS
with spine as (

    select dateadd(day, row_number() over (order by seq4()) - 1, '1950-01-01'::date) as date_day
    from table(generator(rowcount => 31411))

),

days as (

    select
        to_number(to_char(date_day, 'YYYYMMDD'))  as date_key,
        date_day::date                            as full_date,
        year(date_day)                            as year,
        quarter(date_day)                         as quarter,
        month(date_day)                           as month,
        monthname(date_day)                       as month_name,
        day(date_day)                             as day_of_month,
        dayofweekiso(date_day)                    as day_of_week_iso,
        dayofweekiso(date_day) in (6, 7)          as is_weekend,
        -- Año fiscal federal de EE. UU. (octubre-septiembre): así planifica NSF sus convocatorias.
        iff(month(date_day) >= 10, year(date_day) + 1, year(date_day)) as us_fiscal_year
    from spine

)

select * from days
union all
select -1, null, null, null, null, 'Desconocido', null, null, null, null;

-- GOLD · dim_topic  (models/gold/dim_topic.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_TOPIC AS
select
    topic_id        as topic_key,
    topic_name,
    topic_description,
    keywords,
    subfield_id,
    subfield_name,
    field_id,
    field_name,
    domain_id,
    domain_name
from PSET2_DB.SILVER.SLV_TOPICS

union all

select 'UNKNOWN', 'Tema desconocido', null, null, null, null, null, null, null, null;

-- GOLD · dim_funder  (models/gold/dim_funder.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_FUNDER AS
select
    funder_id           as funder_key,
    funder_name,
    country_code,
    is_global_south,
    ror_id,
    homepage_url,
    awards_count,
    works_count
from PSET2_DB.SILVER.SLV_FUNDERS

union all

select 'UNKNOWN', 'Financiador desconocido', null, null, null, null, null, null;

-- GOLD · dim_award  (models/gold/dim_award.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_AWARD AS
select
    award_id            as award_key,
    funder_award_id,
    award_title,
    award_description,
    has_description,
    funder_scheme,
    funding_type,
    doi,
    landing_page_url,
    provenance,
    _loaded_at
from PSET2_DB.SILVER.SLV_AWARDS;

-- GOLD · dim_institution  (models/gold/dim_institution.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_INSTITUTION AS
with awards as (

    select institutions_awarded, _loaded_at
    from PSET2_DB.SILVER.SLV_AWARDS

),

institutions as (

    select
        nullif(split_part(trim(i.value:id::varchar), '/', -1), '')                                   as institution_key,
        nullif(trim(regexp_replace(regexp_replace(i.value:display_name::varchar, '<[^>]+>', ' '), '[[:space:]]+', ' ')), '')                    as institution_name,
        nullif(upper(trim(i.value:country_code::varchar)), '')      as country_code,
        nullif(lower(trim(i.value:type::varchar)), '')              as institution_type,
        nullif(split_part(trim(i.value:ror::varchar), '/', -1), '')                                  as ror_id,
        a._loaded_at
    from awards a,
        lateral flatten(input => a.institutions_awarded) i

)

select *
from institutions
where institution_key is not null
qualify row_number() over (partition by institution_key order by _loaded_at desc) = 1

union all
select 'UNKNOWN', 'Institución desconocida', null, null, null, '1900-01-01'::timestamp_ltz;

-- GOLD · fact_awards  (models/gold/fact_awards.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.FACT_AWARDS AS
with awards as (

    select *
    from PSET2_DB.SILVER.SLV_AWARDS

)

select
    a.award_id                                          as award_key,
    coalesce(f.funder_key, 'UNKNOWN')                   as funder_key,
    coalesce(t.topic_key, 'UNKNOWN')                    as topic_key,
    coalesce(i.institution_key, 'UNKNOWN')              as institution_key,
    iff(a.start_date between '1950-01-01'::date and '2035-12-31'::date,
        to_number(to_char(a.start_date, 'YYYYMMDD')),
        -1)                   as start_date_key,
    iff(a.end_date between '1950-01-01'::date and '2035-12-31'::date,
        to_number(to_char(a.end_date, 'YYYYMMDD')),
        -1)                     as end_date_key,

    a.amount,
    a.currency,
    a.amount_usd,
    iff(a.end_date is not null and a.start_date is not null,
        datediff('day', a.start_date, a.end_date), null) as duration_days,
    a.funded_outputs_count,
    a.institutions_count,
    1                                                   as award_count,

    a.is_amount_invalid,
    a.is_start_date_from_year,
    a.has_inconsistent_dates,
    a._loaded_at

from awards a
inner join PSET2_DB.GOLD.DIM_AWARD da        on da.award_key = a.award_id
left join PSET2_DB.GOLD.DIM_FUNDER f         on f.funder_key = a.funder_id
left join PSET2_DB.GOLD.DIM_TOPIC t          on t.topic_key = a.primary_topic_id
left join PSET2_DB.GOLD.DIM_INSTITUTION i    on i.institution_key = a.lead_institution_id;

-- GOLD · dim_work  (models/gold/dim_work.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_WORK AS
select
    w.work_id                                   as work_key,
    w.doi,
    w.title,
    w.abstract_text,
    w.has_abstract,
    w.publication_year,
    iff(w.publication_date between '1950-01-01'::date and '2035-12-31'::date,
        to_number(to_char(w.publication_date, 'YYYYMMDD')),
        -1)     as publication_date_key,
    w.work_type,
    w.language,
    w.cited_by_count,
    w.fwci,
    w.is_retracted,
    coalesce(t.topic_key, 'UNKNOWN')            as topic_key,
    w.authors_count,
    w._loaded_at
from PSET2_DB.SILVER.SLV_WORKS w
left join PSET2_DB.GOLD.DIM_TOPIC t
    on t.topic_key = w.primary_topic_id;

-- GOLD · dim_author  (models/gold/dim_author.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.DIM_AUTHOR AS
select
    author_id       as author_key,
    author_name,
    orcid,
    _loaded_at
from PSET2_DB.SILVER.SLV_WORK_AUTHORSHIPS

qualify row_number() over (
    partition by author_id
    order by _loaded_at desc, (orcid is null), author_name
) = 1;

-- GOLD · brg_work_author  (models/gold/brg_work_author.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.BRG_WORK_AUTHOR AS
select
    a.work_id                       as work_key,
    a.author_id                     as author_key,
    a.author_position,
    a.is_corresponding,
    a.institution_id,
    a.institution_country_code,
    a._loaded_at
from PSET2_DB.SILVER.SLV_WORK_AUTHORSHIPS a
-- Solo autorías cuyo work y autor existen en Gold (integridad referencial estricta).
inner join PSET2_DB.GOLD.DIM_WORK w   on w.work_key = a.work_id
inner join PSET2_DB.GOLD.DIM_AUTHOR au on au.author_key = a.author_id;

-- GOLD · fact_award_works  (models/gold/fact_award_works.sql)
CREATE OR REPLACE TABLE PSET2_DB.GOLD.FACT_AWARD_WORKS AS
select
    wa.award_id                             as award_key,
    wa.work_id                              as work_key,
    fa.funder_key,
    w.topic_key,
    w.publication_date_key,
    1                                       as funded_work_count,
    wa._loaded_at
from PSET2_DB.SILVER.SLV_WORK_AWARDS wa
inner join PSET2_DB.GOLD.DIM_WORK w      on w.work_key = wa.work_id
inner join PSET2_DB.GOLD.FACT_AWARDS fa   on fa.award_key = wa.award_id;

-- ---------------------------------------------------------------- 4. OBT (spark/jobs/build_obt.py en SQL)
-- GRAIN: una fila por publicacion financiada por un award (award_key + work_key),
-- el mismo de GOLD.FACT_AWARD_WORKS. Los autores se agregan a una fila por work
-- ANTES del join, para no multiplicar filas.
CREATE OR REPLACE TABLE PSET2_DB.OBT.OBT_AWARD_WORK AS
with base as (
    select award_key, work_key, topic_key as work_topic_key, publication_date_key
    from PSET2_DB.GOLD.FACT_AWARD_WORKS
),

autores_por_work as (
    select
        b.work_key,
        array_sort(array_agg(distinct b.author_key))                                       as author_keys,
        max(iff(b.author_position = 'first', b.author_key, null))                          as first_author_key,
        max_by(a.author_name, iff(b.author_position = 'first', b.author_key, null))        as first_author_name,
        array_sort(array_agg(distinct iff(b.is_corresponding, b.author_key, null)))        as corresponding_author_keys,
        array_sort(array_agg(distinct b.institution_country_code))                         as author_country_codes
    from PSET2_DB.GOLD.BRG_WORK_AUTHOR b
    inner join PSET2_DB.GOLD.DIM_AUTHOR a on a.author_key = b.author_key
    where b.work_key in (select work_key from base)
    group by b.work_key
),

unido as (
    select
        base.award_key, base.work_key,
        da.funder_award_id, da.award_title, da.award_description, da.has_description,
        da.funder_scheme, da.funding_type,
        fa.funder_key, f.funder_name, f.country_code as funder_country_code, f.is_global_south,
        fa.institution_key, i.institution_name as lead_institution_name,
        i.country_code as lead_institution_country_code, i.institution_type as lead_institution_type,
        fa.topic_key as award_topic_key, ta.topic_name as award_topic_name, ta.subfield_name as award_subfield_name,
        ta.field_name as award_field_name, ta.domain_name as award_domain_name,
        fa.start_date_key, ds.full_date as award_start_date, ds.year as award_start_year,
        ds.us_fiscal_year as award_us_fiscal_year, fa.end_date_key,
        fa.amount, fa.currency, fa.amount_usd, fa.duration_days, fa.funded_outputs_count, fa.institutions_count,
        fa.is_amount_invalid, fa.has_inconsistent_dates,
        w.doi as work_doi, w.title as work_title, w.abstract_text, w.has_abstract, w.publication_year,
        base.publication_date_key, dp.full_date as publication_date, w.work_type, w.language,
        w.cited_by_count, w.fwci, w.is_retracted, w.authors_count,
        base.work_topic_key, tw.topic_name as work_topic_name, tw.subfield_name as work_subfield_name,
        tw.field_name as work_field_name, tw.domain_name as work_domain_name,
        ap.author_keys, ap.first_author_key, ap.first_author_name,
        ap.corresponding_author_keys, ap.author_country_codes,
        -- marcas de huerfanos: NULL solo si el join obligatorio no encontro pareja
        fa.award_key as _m_fact_awards, da.award_key as _m_dim_award, f.funder_key as _m_dim_funder,
        i.institution_key as _m_dim_institution, w.work_key as _m_dim_work, ta.topic_key as _m_award_topic,
        tw.topic_key as _m_work_topic, ds.date_key as _m_start_date, dp.date_key as _m_publication_date
    from base
    left join PSET2_DB.GOLD.FACT_AWARDS fa      on fa.award_key = base.award_key
    left join PSET2_DB.GOLD.DIM_AWARD da        on da.award_key = base.award_key
    left join PSET2_DB.GOLD.DIM_FUNDER f        on f.funder_key = fa.funder_key
    left join PSET2_DB.GOLD.DIM_INSTITUTION i   on i.institution_key = fa.institution_key
    left join PSET2_DB.GOLD.DIM_WORK w          on w.work_key = base.work_key
    left join PSET2_DB.GOLD.DIM_TOPIC ta        on ta.topic_key = fa.topic_key
    left join PSET2_DB.GOLD.DIM_TOPIC tw        on tw.topic_key = base.work_topic_key
    left join PSET2_DB.GOLD.DIM_DATE ds         on ds.date_key = fa.start_date_key
    left join PSET2_DB.GOLD.DIM_DATE dp         on dp.date_key = base.publication_date_key
    left join autores_por_work ap               on ap.work_key = base.work_key
),

textos as (
    select
        u.*,
        nullif(trim(award_title), '')       as _t_award,
        nullif(trim(award_description), '') as _b_award,
        nullif(trim(work_title), '')        as _t_work,
        nullif(trim(abstract_text), '')     as _b_work
    from unido u
)

select
    award_key, work_key,
    1 as label,
    funder_award_id, award_title, award_description, has_description, funder_scheme, funding_type,
    funder_key, funder_name, funder_country_code, is_global_south,
    institution_key, lead_institution_name, lead_institution_country_code, lead_institution_type,
    award_topic_key, award_topic_name, award_subfield_name, award_field_name, award_domain_name,
    start_date_key, award_start_date, award_start_year, award_us_fiscal_year, end_date_key,
    amount, currency, amount_usd, duration_days, funded_outputs_count, institutions_count,
    is_amount_invalid, has_inconsistent_dates,
    work_doi, work_title, abstract_text, has_abstract, publication_year,
    publication_date_key, publication_date, work_type, language,
    cited_by_count, fwci, is_retracted, authors_count,
    work_topic_key, work_topic_name, work_subfield_name, work_field_name, work_domain_name,
    coalesce(author_keys, array_construct())               as author_keys,
    coalesce(array_size(author_keys), 0)                   as linked_authors_count,
    first_author_key, first_author_name,
    coalesce(corresponding_author_keys, array_construct()) as corresponding_author_keys,
    coalesce(author_country_codes, array_construct())      as author_country_codes,
    -- 'Titulo. Cuerpo' (si el titulo ya termina en puntuacion, solo un espacio)
    case when _t_award is null then _b_award when _b_award is null then _t_award
         when regexp_like(_t_award, '.*[.!?:;…]$') then _t_award || ' ' || _b_award
         else _t_award || '. ' || _b_award end           as award_text,
    case when _t_work is null then _b_work when _b_work is null then _t_work
         when regexp_like(_t_work, '.*[.!?:;…]$') then _t_work || ' ' || _b_work
         else _t_work || '. ' || _b_work end             as work_text,
    current_timestamp()                                    as _obt_built_at,
    -- huerfanos (se revisan abajo y se quitan al final)
    (iff(_m_fact_awards is null, 1, 0) + iff(_m_dim_award is null, 1, 0) + iff(_m_dim_funder is null, 1, 0)
     + iff(_m_dim_institution is null, 1, 0) + iff(_m_dim_work is null, 1, 0) + iff(_m_award_topic is null, 1, 0)
     + iff(_m_work_topic is null, 1, 0) + iff(_m_start_date is null, 1, 0) + iff(_m_publication_date is null, 1, 0))
                                                           as _joins_sin_pareja
from textos;

-- ---------------------------------------------------------------- 5. Validacion (las garantias del job de Spark)
-- Las tres columnas de la derecha deben ser: obt_filas = fact_filas = pares_distintos, y 0 huerfanos.
SELECT
    (SELECT COUNT(*) FROM PSET2_DB.GOLD.FACT_AWARD_WORKS)                         AS fact_filas,
    COUNT(*)                                                                       AS obt_filas,
    COUNT(DISTINCT award_key, work_key)                                            AS pares_distintos,
    COUNT_IF(_joins_sin_pareja > 0)                                                AS filas_con_huerfanos
FROM PSET2_DB.OBT.OBT_AWARD_WORK;

ALTER TABLE PSET2_DB.OBT.OBT_AWARD_WORK DROP COLUMN _joins_sin_pareja;

-- Listo. Ahora corre psets/proyecto/modelado/exportar_obt.sql y descarga el CSV.
