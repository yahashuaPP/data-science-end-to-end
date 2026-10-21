-- Correr en Snowsight (PSET2_DB) y descargar el resultado como CSV:
-- guardalo junto a este script como obt_award_work_muestra.csv
-- Muestra de 50 000 pares: la OBT completa (1,88M filas con textos largos)
-- es demasiado para un CSV. Los textos van como longitud, no completos.
select
    award_key, work_key, label,
    funder_name, funder_country_code, is_global_south, funding_type,
    award_field_name, award_domain_name, work_field_name, work_domain_name,
    award_start_year, publication_year,
    amount_usd, is_amount_invalid, duration_days, funded_outputs_count,
    has_description, has_abstract,
    length(award_text) as award_text_len,
    length(work_text)  as work_text_len,
    cited_by_count, fwci, authors_count, linked_authors_count, language, work_type
from OBT.OBT_AWARD_WORK
sample (50000 rows);
