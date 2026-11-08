"""Preparacion de los datos de GrantMatch, lista para produccion.

Las features que decidi en features_proyecto.ipynb viven aqui, dentro de
funciones. Desde la Semana 10, los notebooks de experimentos importan este
archivo en vez de copiar el codigo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que corre de principio a fin, sin notebook:

    python3 pipeline_proyecto.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pares (award, publicacion) con label: los reales de la OBT y los
# negativos armados al azar. Lo escribe features_proyecto.ipynb.
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "processed" / "pares_proyecto.csv"

OBJETIVO = "label"
# Solo columnas que existen al recomendar un award a un investigador.
# Fuera: funded_outputs_count (se sabe despues del financiamiento),
# publication_year (en un par real la publicacion sale despues del award,
# asi que delataria el label), fwci, identificadores y textos.
NUMERICAS = ["amount_usd", "duration_days", "authors_count", "cited_by_count"]
CATEGORICAS = ["funding_type", "award_domain_name", "work_domain_name"]
YA_LISTAS = ["mismo_campo", "funder_en_pais_autor", "is_global_south"]

COLUMNA_FECHA = "award_start_year"
PROPORCION_PRUEBA = 0.2


def agregar_features(datos):
    datos = datos.copy()
    # Feature 1: el award y la publicacion son del mismo campo.
    datos["mismo_campo"] = (
        datos["award_field_name"].notna()
        & (datos["award_field_name"] == datos["work_field_name"])
    ).astype(int)
    # Feature 2: el pais del funder esta entre los paises de los autores.
    paises_autores = datos["author_country_codes"].fillna("").astype(str).str.upper().str.split("|")
    pais_funder = datos["funder_country_code"].fillna("").astype(str).str.upper()
    datos["funder_en_pais_autor"] = [
        int(bool(pais) and pais in paises) for pais, paises in zip(pais_funder, paises_autores)
    ]
    # is_global_south llega como texto o booleano desde Snowflake.
    datos["is_global_south"] = datos["is_global_south"].map(
        {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}
    ).astype(float)
    return datos


def anio_de_corte(datos):
    """Ano desde el cual los awards van a prueba: el percentil 80 de
    award_start_year (uno mas si con ese corte la prueba pasa del 40 %)."""
    anos = datos[COLUMNA_FECHA].dropna().astype(int)
    corte = int(anos.quantile(1 - PROPORCION_PRUEBA))
    if (anos >= corte).mean() > 0.4:
        corte += 1
    return corte


def cargar_datos(ruta=RUTA_DATOS):
    """Los pares con sus features, sin los awards que no tienen fecha de
    inicio. Lo usa la validacion cruzada (Semana 11.2), que reparte todas
    las filas en pliegues en vez de usar una sola division."""
    datos = agregar_features(pd.read_csv(ruta))
    return datos[datos[COLUMNA_FECHA].notna()].reset_index(drop=True)


def cargar_y_dividir(ruta=RUTA_DATOS):
    """Division por fecha: entrenamiento con los awards que empezaron antes
    del ano de corte, prueba con los demas. Un award y su par negativo
    comparten el ano, asi que caen siempre del mismo lado."""
    datos = cargar_datos(ruta)
    en_prueba = datos[COLUMNA_FECHA].astype(int) >= anio_de_corte(datos)
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
    y = datos[OBJETIVO]
    return X[~en_prueba], X[en_prueba], y[~en_prueba], y[en_prueba]


def construir_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
        ]), YA_LISTAS),
    ])


# Esto corre con "python3 pipeline_proyecto.py", pero no cuando un
# notebook hace "from pipeline_proyecto import ...".
if __name__ == "__main__":
    print("Ano de corte:", anio_de_corte(pd.read_csv(RUTA_DATOS)))
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    print("Proporcion de pares reales  entrenamiento:", round(y_ent.mean(), 3), " prueba:", round(y_prueba.mean(), 3))
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))
