"""Pruebas de pipeline_proyecto.py. Se corren desde la raiz del fork, con el
mismo comando que usa GitHub Actions:

    pytest psets/proyecto -v

No leen data/: en GitHub Actions ese archivo no existe. Cada prueba arma sus
propios pares (award, publicacion) escritos a mano, con valores que se pueden
calcular de cabeza.
"""

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    YA_LISTAS,
    agregar_features,
    anio_de_corte,
    cargar_y_dividir,
    construir_preparacion,
)


def pares_de_prueba():
    """Diez pares: cinco awards (uno por ano, de 2015 a 2019), cada uno con
    su publicacion real (label 1) y una al azar (label 0)."""
    filas = []
    for i, anio in enumerate(range(2015, 2020)):
        award = {
            "award_key": f"A{i}",
            "award_start_year": anio,
            "amount_usd": 50_000.0 + 10_000 * i,
            "duration_days": 365.0 * (1 + i % 3),
            "funding_type": ["grant", "fellowship"][i % 2],
            "award_field_name": "Physics",
            "award_domain_name": "Physical Sciences",
            "funder_country_code": "US",
            "is_global_south": "false",
        }
        filas.append({**award, OBJETIVO: 1, "work_field_name": "Physics", "work_domain_name": "Physical Sciences",
                      "author_country_codes": "US|DE", "authors_count": 3, "cited_by_count": 10 + i})
        filas.append({**award, OBJETIVO: 0, "work_field_name": "Medicine", "work_domain_name": "Health Sciences",
                      "author_country_codes": "BR", "authors_count": 8, "cited_by_count": 2 * i})
    return pd.DataFrame(filas)


def test_mismo_campo_calculado_a_mano():
    pares = pd.DataFrame({
        "award_field_name": ["Physics", "Physics", np.nan],
        "work_field_name": ["Physics", "Medicine", np.nan],
        "funder_country_code": ["US", "US", "US"],
        "author_country_codes": ["US", "US", "US"],
        "is_global_south": [False, False, False],
    })
    # Mismo campo, campo distinto, y los dos vacios: dos NaN no son "el mismo
    # campo", porque no se sabe el campo del award.
    assert agregar_features(pares)["mismo_campo"].tolist() == [1, 0, 0]


def test_funder_en_pais_autor_calculado_a_mano():
    pares = pd.DataFrame({
        "award_field_name": ["Physics"] * 5,
        "work_field_name": ["Physics"] * 5,
        "funder_country_code": ["us", "US", "US", np.nan, "EC"],
        "author_country_codes": ["DE|US", "BR|AR", np.nan, "US", "ec"],
        "is_global_south": [False] * 5,
    })
    # Las mayusculas no importan; sin paises de autores o sin pais del
    # funder la feature es 0, no un error.
    assert agregar_features(pares)["funder_en_pais_autor"].tolist() == [1, 0, 0, 0, 1]


def test_is_global_south_queda_en_0_y_1():
    pares = pares_de_prueba().head(4)
    pares["is_global_south"] = ["true", "FALSE", True, False]
    assert agregar_features(pares)["is_global_south"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_agregar_features_no_modifica_la_entrada():
    pares = pares_de_prueba()
    columnas_antes = list(pares.columns)
    agregar_features(pares)
    assert list(pares.columns) == columnas_antes


def test_el_objetivo_no_llega_al_modelo(tmp_path):
    assert OBJETIVO not in NUMERICAS + CATEGORICAS + YA_LISTAS
    ruta = tmp_path / "pares.csv"
    pares_de_prueba().to_csv(ruta, index=False)
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    assert OBJETIVO not in X_ent.columns
    assert OBJETIVO not in X_prueba.columns
    # Tampoco las columnas que se conocen despues del financiamiento.
    for columna in ["funded_outputs_count", "publication_year", "award_key"]:
        assert columna not in X_ent.columns


def test_division_por_anio_no_mira_el_futuro(tmp_path):
    pares = pares_de_prueba()
    ruta = tmp_path / "pares.csv"
    pares.to_csv(ruta, index=False)
    corte = anio_de_corte(pares)
    # 10 pares, 2 por ano de 2015 a 2019: el percentil 80 da 2018 y con ese
    # corte la prueba tiene 4 de 10 pares (no pasa del 40 %).
    assert corte == 2018
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    anios_ent = pares.loc[X_ent.index, "award_start_year"]
    anios_prueba = pares.loc[X_prueba.index, "award_start_year"]
    assert anios_ent.max() < corte <= anios_prueba.min()
    # Un award y su par negativo caen del mismo lado: ningun award esta en
    # entrenamiento y en prueba a la vez.
    awards_ent = set(pares.loc[X_ent.index, "award_key"])
    awards_prueba = set(pares.loc[X_prueba.index, "award_key"])
    assert awards_ent.isdisjoint(awards_prueba)
    assert len(X_ent) + len(X_prueba) == len(pares)


def test_preparacion_acepta_una_categoria_nueva():
    pares = agregar_features(pares_de_prueba())
    X = pares[NUMERICAS + CATEGORICAS + YA_LISTAS]
    preparacion = construir_preparacion()
    columnas_entrenamiento = preparacion.fit_transform(X).shape[1]

    nuevo = X.head(1).copy()
    nuevo["funding_type"] = "prize"          # no aparece en el entrenamiento
    nuevo["amount_usd"] = np.nan             # y le falta el monto
    transformado = preparacion.transform(nuevo)
    assert transformado.shape == (1, columnas_entrenamiento)
    assert not np.isnan(transformado).any()
    # La categoria nueva queda en cero en todas las columnas de funding_type.
    nombres = list(preparacion.get_feature_names_out())
    indices = [i for i, nombre in enumerate(nombres) if "funding_type" in nombre]
    assert transformado[0, indices].sum() == 0
