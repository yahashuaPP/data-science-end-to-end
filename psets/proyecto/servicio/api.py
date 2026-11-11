"""Servicio de prediccion de GrantMatch: el modelo final detras de una API.

Quien llama (el lote nocturno, un tablero, otra aplicacion) manda un par
(award, publicacion) como lo conoce: los datos de la convocatoria y los del
perfil del investigador. El servicio calcula las features con las mismas
funciones de modelado/pipeline_proyecto.py con que se entreno, y devuelve la
probabilidad de que el award financie ese trabajo.

Desde psets/proyecto/, despues de python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000

La documentacion interactiva queda en el puerto 8000, ruta /docs.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_features

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_INFO = CARPETA / "modelo_info.json"

# Desde que probabilidad se recomienda el award. Con los costos de la
# Semana 11 (perder un award pertinente cuesta 10, una recomendacion de mas
# cuesta 1) el umbral es bajo: pon aqui el de menor costo que dio
# validacion_proyecto.ipynb, o pasalo como variable de entorno.
UMBRAL = float(os.environ.get("UMBRAL", "0.10"))

# El modelo se carga una sola vez, al arrancar.
modelo = None
version_modelo = "desconocida"
if RUTA_MODELO.exists():
    modelo = joblib.load(RUTA_MODELO)
    if RUTA_INFO.exists():
        version_modelo = json.loads(RUTA_INFO.read_text(encoding="utf-8"))["run_id"]

app = FastAPI(title="GrantMatch: probabilidad de financiamiento")


class Par(BaseModel):
    """Un award y una publicacion, con lo que se sabe al recomendar.

    Son obligatorios los cuatro datos con que se calculan mismo_campo y
    funder_en_pais_autor: si faltaran, las features saldrian 0 sin ningun
    error. El resto puede llegar vacio: la preparacion lo imputa.
    """
    # La convocatoria (award)
    award_field_name: str
    funder_country_code: str = Field(min_length=2, max_length=2)
    amount_usd: float | None = Field(default=None, ge=0)
    duration_days: float | None = Field(default=None, ge=0)
    funding_type: str | None = None
    award_domain_name: str | None = None
    is_global_south: bool | None = None
    # El perfil del investigador (su publicacion)
    work_field_name: str
    author_country_codes: list[str]
    work_domain_name: str | None = None
    authors_count: int | None = Field(default=None, ge=1)
    cited_by_count: int | None = Field(default=None, ge=0)


def preparar_entrada(par):
    """El par como DataFrame de una fila, con las features del entrenamiento.
    author_country_codes llega como lista y la OBT lo exporta como "US|DE"."""
    fila = par.model_dump()
    fila["author_country_codes"] = "|".join(fila["author_country_codes"])
    return agregar_features(pd.DataFrame([fila]))


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.post("/predecir")
def predecir(par: Par):
    if modelo is None:
        raise HTTPException(status_code=503, detail="No hay modelo cargado: corre python servicio/exportar_modelo.py")
    pares = preparar_entrada(par)
    probabilidad = float(modelo.predict_proba(pares[NUMERICAS + CATEGORICAS + YA_LISTAS])[0, 1])
    return {
        "probabilidad_financiamiento": round(probabilidad, 4),
        "recomendar": probabilidad >= UMBRAL,
        "version_modelo": version_modelo,
    }
