"""Ejercicio 3 (Semana 9, Sesion 1): dos features para GrantMatch.

Entrada: la muestra de la OBT (OBT.OBT_AWARD_WORK) que exporta
semana-08-sesion-02/plantilla-eda/exportar_obt.sql.
Salida: pares_con_features.csv, en esta misma carpeta, que usan el
pipeline (Semana 9, Sesion 2) y los modelos (Semana 10).

En la OBT todas las filas son financiamientos reales (label = 1). Para
tener algo que predecir, este script arma pares negativos: a cada award
de la muestra le asigna una publicacion que NO financio (label = 0).
"""

from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RUTA_OBT = CARPETA.parent.parent / "semana-08-sesion-02" / "plantilla-eda" / "obt_award_work_muestra.csv"
RUTA_SALIDA = CARPETA / "pares_con_features.csv"

OBJETIVO = "label"
SEMILLA = 42

# Columnas que describen el award (la convocatoria) y la publicacion (el
# perfil del investigador). Un par negativo toma las del award de una fila
# y las de la publicacion de otra.
COLUMNAS_AWARD = [
    "award_key", "funder_name", "funder_country_code", "is_global_south", "funding_type",
    "award_field_name", "award_domain_name", "award_start_year", "amount_usd",
    "is_amount_invalid", "duration_days", "funded_outputs_count", "has_description", "award_text_len",
]
COLUMNAS_PUBLICACION = [
    "work_key", "work_field_name", "work_domain_name", "publication_year", "has_abstract",
    "work_text_len", "cited_by_count", "fwci", "authors_count", "linked_authors_count",
    "language", "work_type", "author_country_codes",
]

datos = pd.read_csv(RUTA_OBT)
datos.columns = datos.columns.str.lower()  # Snowflake exporta en MAYUSCULAS
print("Muestra de la OBT:", datos.shape)
print("Valores de label en la OBT:", datos[OBJETIVO].value_counts().to_dict())

# Paso 0: pares negativos. Para cada par real se elige al azar la
# publicacion de otra fila, cuidando que ese award no la haya financiado.
positivos = datos[COLUMNAS_AWARD + COLUMNAS_PUBLICACION].copy()
positivos[OBJETIVO] = 1

pares_reales = set(zip(positivos["award_key"], positivos["work_key"]))
generador = np.random.default_rng(SEMILLA)
indices_publicacion = generador.permutation(len(positivos))
# Rota la permutacion hasta que ningun par negativo sea un par real.
for _ in range(10):
    choque = np.array([
        (a, w) in pares_reales
        for a, w in zip(positivos["award_key"].values, positivos["work_key"].values[indices_publicacion])
    ])
    if not choque.any():
        break
    indices_publicacion[choque] = generador.permutation(indices_publicacion[choque])

negativos = pd.concat([
    positivos[COLUMNAS_AWARD].reset_index(drop=True),
    positivos[COLUMNAS_PUBLICACION].iloc[indices_publicacion].reset_index(drop=True),
], axis=1)
negativos[OBJETIVO] = 0
choques = sum((a, w) in pares_reales for a, w in zip(negativos["award_key"], negativos["work_key"]))
negativos = negativos[[(a, w) not in pares_reales for a, w in zip(negativos["award_key"], negativos["work_key"])]]
print(f"Pares negativos descartados por ser reales: {choques}")

pares = pd.concat([positivos, negativos], ignore_index=True)
print("Pares con label:", pares[OBJETIVO].value_counts().to_dict())

# Feature 1 (hallazgo del EDA, Semana 8): el award y la publicacion son
# del mismo campo de investigacion. Se conoce al momento de recomendar:
# el campo del award viene de la convocatoria y el de la publicacion del
# perfil del investigador.
pares["mismo_campo"] = (
    pares["award_field_name"].notna()
    & (pares["award_field_name"] == pares["work_field_name"])
).astype(int)

# Feature 2 (conocimiento del dominio): el funder es del mismo pais que
# alguno de los autores. La mayoria de las agencias financian sobre todo a
# investigadores de su propio pais. author_country_codes llega como texto
# "US|DE|...".
paises_autores = pares["author_country_codes"].fillna("").astype(str).str.upper().str.split("|")
pais_funder = pares["funder_country_code"].fillna("").astype(str).str.upper()
pares["funder_en_pais_autor"] = [
    int(bool(pais) and pais in paises) for pais, paises in zip(pais_funder, paises_autores)
]

# Columnas booleanas como 0/1, para que el pipeline las pueda pasar tal cual.
for columna in ["is_global_south", "is_amount_invalid", "has_description", "has_abstract"]:
    pares[columna] = pares[columna].map({True: 1, False: 0, "true": 1, "false": 0, "TRUE": 1, "FALSE": 0}).astype(float)

# Evidencia: el objetivo promedio segun cada feature binaria.
print()
print("Proporcion de pares reales segun mismo_campo:")
print(pares.groupby("mismo_campo")[OBJETIVO].agg(["mean", "size"]).round(3))
print()
print("Proporcion de pares reales segun funder_en_pais_autor:")
print(pares.groupby("funder_en_pais_autor")[OBJETIVO].agg(["mean", "size"]).round(3))
print()
print("Correlacion de cada feature con label:")
print(pares[["mismo_campo", "funder_en_pais_autor", OBJETIVO]].corr()[OBJETIVO].round(3))

pares.to_csv(RUTA_SALIDA, index=False)
print()
print(f"Guardado {RUTA_SALIDA.name}: {pares.shape}")
print(pares.head())
