"""Ejercicio 3 (Semana 9, Sesion 2): el pipeline de GrantMatch.

Lee pares_con_features.csv (lo escribe
semana-09-sesion-01/plantilla-features/features_proyecto.py), divide por
fecha y prepara las columnas. Los scripts de la Semana 10 importan de aqui
cargar_datos, dividir y preparacion, para usar la misma division y la misma
preparacion.

    python3 pipeline_proyecto.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CARPETA = Path(__file__).resolve().parent
RUTA_PARES = CARPETA.parent.parent / "semana-09-sesion-01" / "plantilla-features" / "pares_con_features.csv"

OBJETIVO = "label"
# Solo columnas que existen al recomendar un award a un investigador.
# Fuera: funded_outputs_count (cuantas publicaciones financio el award: se
# sabe despues), publication_year (las publicaciones de un par real son
# resultado del award, asi que van despues de el), fwci y textos.
NUMERICAS = ["amount_usd", "duration_days", "authors_count", "cited_by_count"]
CATEGORICAS = ["funding_type", "award_domain_name", "work_domain_name"]
YA_LISTAS = ["mismo_campo", "funder_en_pais_autor", "is_global_south"]
COLUMNA_FECHA = "award_start_year"
PROPORCION_PRUEBA = 0.2


def cargar_datos(ruta=RUTA_PARES):
    datos = pd.read_csv(ruta)
    return datos[datos[COLUMNA_FECHA].notna()].copy()


def dividir(datos):
    """Division por fecha: entrenamiento con los awards que empezaron antes
    del ano de corte, prueba con los demas. Un award y su par negativo caen
    siempre del mismo lado, porque comparten el ano."""
    anos = datos[COLUMNA_FECHA].astype(int)
    corte = int(anos.quantile(1 - PROPORCION_PRUEBA))
    en_prueba = anos >= corte
    if en_prueba.mean() > 0.4:  # muchos awards en el mismo ano: corre el corte uno
        corte += 1
        en_prueba = anos >= corte
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
    y = datos[OBJETIVO]
    return X[~en_prueba], X[en_prueba], y[~en_prueba], y[en_prueba], corte


def preparacion():
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


if __name__ == "__main__":
    datos = cargar_datos()
    X_ent, X_prueba, y_ent, y_prueba, corte = dividir(datos)
    print(f"Ano de corte: awards desde {corte} van a prueba")
    print("Filas de entrenamiento:", len(X_ent), " de prueba:", len(X_prueba))
    print("Proporcion de pares reales  entrenamiento:", round(y_ent.mean(), 3), " prueba:", round(y_prueba.mean(), 3))

    prep = preparacion()
    X_ent_listo = prep.fit_transform(X_ent)
    X_prueba_listo = prep.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(prep.get_feature_names_out()))
