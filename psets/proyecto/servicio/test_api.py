"""Pruebas del servicio de GrantMatch. Desde la raiz del fork:

    pytest psets/proyecto -v

No usan servicio/modelo.joblib (en GitHub Actions no existe): reemplazan el
modelo por uno pequeno que responde 0.25 para cualquier par. Revisan el
contrato de la API, no la calidad del modelo.
"""

import json

import pandas as pd
import pytest
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
api.version_modelo = "7"  # una version inventada del registro
cliente = TestClient(api.app)


@pytest.fixture(autouse=True)
def registro_temporal(tmp_path, monkeypatch):
    """Cada prueba escribe sus predicciones en un archivo temporal (no en
    monitoreo/predicciones.jsonl) y empieza con las metricas en cero."""
    monkeypatch.setattr(api, "RUTA_REGISTRO", tmp_path / "predicciones.jsonl")
    monkeypatch.setattr(api, "metricas", api.Metricas())
    return tmp_path / "predicciones.jsonl"


def test_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7"}


def test_par_valido():
    respuesta = cliente.post("/predecir", json=PAR)
    assert respuesta.status_code == 200
    # 0.25 alcanza el umbral por defecto (0.10)
    cuerpo = respuesta.json()
    assert len(cuerpo.pop("id_prediccion")) == 32
    assert cuerpo == {"probabilidad_financiamiento": 0.25, "recomendar": True, "version_modelo": "7"}


def test_par_con_datos_opcionales_vacios():
    par = {k: PAR[k] for k in ["work_field_name", "author_country_codes"]}
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


def test_cada_prediccion_dice_que_version_la_calculo():
    # Semana 13: quien recibe una prediccion tiene que poder saber que
    # version del registro la calculo, sin preguntarle a MLflow.
    respuesta = cliente.post("/predecir", json=PAR)
    assert respuesta.json()["version_modelo"] == "7"
    assert cliente.get("/salud").json()["version_modelo"] == "7"


def test_cada_prediccion_queda_registrada(registro_temporal):
    # Semana 13.2: el registro guarda lo necesario para unir la prediccion
    # con la verdad cuando llegue (los identificadores del par).
    par = {**PAR, "award_key": "G123", "work_key": "W456"}
    respuesta = cliente.post("/predecir", json=par).json()
    lineas = [json.loads(linea) for linea in registro_temporal.read_text().splitlines()]
    assert len(lineas) == 1
    assert lineas[0]["id_prediccion"] == respuesta["id_prediccion"]
    assert lineas[0]["version_modelo"] == "7"
    assert lineas[0]["entrada"]["award_key"] == "G123"
    assert lineas[0]["entrada"]["work_key"] == "W456"
    assert lineas[0]["probabilidad_financiamiento"] == 0.25
    assert lineas[0]["recomendar"] is True


def test_un_par_rechazado_no_se_registra(registro_temporal):
    cliente.post("/predecir", json={**PAR, "amount_usd": -5})
    assert not registro_temporal.exists()


def test_metricas_separan_errores_de_validacion_y_de_servidor(monkeypatch):
    cliente.post("/predecir", json=PAR)                       # 200
    cliente.post("/predecir", json={"award_field_name": "X"})  # 422: faltan datos

    class ModeloRoto:
        def predict_proba(self, X):
            raise RuntimeError("el modelo fallo")

    monkeypatch.setattr(api, "modelo", ModeloRoto())
    roto = TestClient(api.app, raise_server_exceptions=False)
    assert roto.post("/predecir", json=PAR).status_code == 500

    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 3
    assert resumen["errores_validacion"] == 1
    assert resumen["errores_servidor"] == 1
    assert resumen["latencia_p95_ms"] >= resumen["latencia_p50_ms"] > 0


def test_la_alerta_de_deriva_se_enciende_lejos_de_la_referencia(monkeypatch):
    # Semana 14: una referencia en la que casi todos los pares tienen el
    # financiador en el pais de un autor, y una ventana llena de pares sin el.
    referencia = {
        "funder_en_pais_autor": {"categorias": [0, 1], "proporciones": [0.1, 0.9]},
        "funder_country_code": {"categorias": ["US", "otro"], "proporciones": [0.5, 0.5]},
        "cited_by_count": {"cortes": [10], "proporciones": [0.5, 0.5]},
    }
    monkeypatch.setattr(api, "monitor", api.MonitorDeDeriva(referencia, ventana=4, umbral=0.10))
    lejano = {**PAR, "author_country_codes": ["DE"], "funder_country_code": "BR", "cited_by_count": 50}
    for _ in range(3):
        cliente.post("/predecir", json=lejano)
    assert cliente.get("/deriva").json()["alerta"] is False      # la ventana todavia no esta llena
    cliente.post("/predecir", json=lejano)
    estado = cliente.get("/deriva").json()
    assert estado["pares_en_ventana"] == 4
    assert estado["psi"]["funder_en_pais_autor"] > 0.10
    assert estado["alerta"] is True


def test_sin_deriva_no_hay_alerta(monkeypatch):
    referencia = {
        "funder_en_pais_autor": {"categorias": [0, 1], "proporciones": [0.0, 1.0]},
        "funder_country_code": {"categorias": ["US", "otro"], "proporciones": [1.0, 0.0]},
        "cited_by_count": {"cortes": [10], "proporciones": [0.0, 1.0]},
    }
    monkeypatch.setattr(api, "monitor", api.MonitorDeDeriva(referencia, ventana=2, umbral=0.10))
    for _ in range(2):
        cliente.post("/predecir", json=PAR)   # US, autores de US y DE, 12 citas
    assert cliente.get("/deriva").json()["alerta"] is False
