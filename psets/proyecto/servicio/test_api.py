"""Pruebas del servicio de GrantMatch. Desde la raiz del fork:

    pytest psets/proyecto -v

No usan servicio/modelo.joblib (en GitHub Actions no existe): reemplazan el
modelo por uno pequeno que responde 0.25 para cualquier par. Revisan el
contrato de la API, no la calidad del modelo.
"""

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, construir_preparacion
from servicio import api

PAR = {
    "award_field_name": "Physics and Astronomy",
    "funder_country_code": "US",
    "amount_usd": 250000.0,
    "duration_days": 1095,
    "funding_type": "grant",
    "award_domain_name": "Physical Sciences",
    "is_global_south": False,
    "work_field_name": "Physics and Astronomy",
    "author_country_codes": ["US", "DE"],
    "work_domain_name": "Physical Sciences",
    "authors_count": 4,
    "cited_by_count": 12,
}


def modelo_de_prueba():
    """La preparacion real con un modelo que responde la proporcion con que
    se entreno: un par real de cuatro, 0.25 para todos."""
    pares = pd.concat([api.preparar_entrada(api.Par(**PAR))] * 4, ignore_index=True)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyClassifier(strategy="prior"))])
    modelo.fit(pares[NUMERICAS + CATEGORICAS + YA_LISTAS], [1, 0, 0, 0])
    return modelo


api.modelo = modelo_de_prueba()
api.version_modelo = "abc123"
cliente = TestClient(api.app)


def test_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "abc123"}


def test_par_valido():
    respuesta = cliente.post("/predecir", json=PAR)
    assert respuesta.status_code == 200
    # 0.25 alcanza el umbral por defecto (0.10)
    assert respuesta.json() == {"probabilidad_financiamiento": 0.25, "recomendar": True, "version_modelo": "abc123"}


def test_par_con_datos_opcionales_vacios():
    par = {k: PAR[k] for k in ["award_field_name", "funder_country_code", "work_field_name", "author_country_codes"]}
    respuesta = cliente.post("/predecir", json=par)
    assert respuesta.status_code == 200
    assert 0 <= respuesta.json()["probabilidad_financiamiento"] <= 1


def test_par_sin_campo_de_la_publicacion_se_rechaza():
    par = PAR.copy()
    par.pop("work_field_name")
    respuesta = cliente.post("/predecir", json=par)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "work_field_name"]


def test_paises_como_texto_se_rechazan():
    # La OBT exporta los paises como "US|DE", pero el servicio pide una lista:
    # un texto suelto se rechaza en vez de leerse como un solo pais.
    par = {**PAR, "author_country_codes": "US|DE"}
    assert cliente.post("/predecir", json=par).status_code == 422


def test_monto_negativo_se_rechaza():
    assert cliente.post("/predecir", json={**PAR, "amount_usd": -5}).status_code == 422


def test_el_servicio_calcula_las_features_como_el_entrenamiento():
    # mismo_campo y funder_en_pais_autor salen de agregar_features de
    # modelado/pipeline_proyecto.py, la misma funcion del entrenamiento.
    pares = api.preparar_entrada(api.Par(**PAR))
    assert pares.loc[0, "mismo_campo"] == 1
    assert pares.loc[0, "funder_en_pais_autor"] == 1
    otro = api.Par(**{**PAR, "work_field_name": "Medicine", "author_country_codes": ["BR"]})
    pares = api.preparar_entrada(otro)
    assert pares.loc[0, "mismo_campo"] == 0
    assert pares.loc[0, "funder_en_pais_autor"] == 0
