"""Servicio de prediccion de GrantMatch: el modelo final detras de una API.

Quien llama (el lote nocturno, un tablero, otra aplicacion) manda un par
(award, publicacion) como lo conoce: los datos de la convocatoria y los del
perfil del investigador. El servicio calcula las features con las mismas
funciones de modelado/pipeline_proyecto.py con que se entreno, y devuelve la
probabilidad de que el award financie ese trabajo.

Desde psets/proyecto/, despues de python -m servicio.exportar_modelo:

    uvicorn servicio.api:app --port 8000

La documentacion interactiva queda en el puerto 8000, ruta /docs.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_features
from servicio.deriva import MonitorDeDeriva

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_VERSION = CARPETA / "modelo_version.txt"
# Cada prediccion queda en una linea de este archivo (Semana 13): es lo que
# despues se une con la verdad en monitoreo/desempeno.py. No se sube a git.
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", CARPETA.parent / "monitoreo" / "predicciones.jsonl"))
# La referencia de deriva (Semana 14): la escribe monitoreo/simular_deriva.py.
RUTA_REFERENCIA = CARPETA.parent / "monitoreo" / "referencia_deriva.json"

# Desde que probabilidad se recomienda el award. Con los costos de la
# Semana 11 (perder un award pertinente cuesta 10, una recomendacion de mas
# cuesta 1) el umbral es bajo: pon aqui el de menor costo que dio
# validacion_proyecto.ipynb, o pasalo como variable de entorno.
UMBRAL = float(os.environ.get("UMBRAL", "0.10"))

# El modelo se carga una sola vez, al arrancar.
modelo = None
# La version del registro de MLflow que se exporto con el modelo
# (servicio/exportar_modelo.py): cada respuesta dice cual la calculo.
version_modelo = "desconocida"
if RUTA_MODELO.exists():
    modelo = joblib.load(RUTA_MODELO)
    if RUTA_VERSION.exists():
        version_modelo = RUTA_VERSION.read_text(encoding="utf-8").strip()



class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir. Se reinician cuando se reinicia el servicio."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        hay = len(self.latencias_ms) > 0
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(1 for c in self.codigos if c == 422),
            "errores_servidor": sum(1 for c in self.codigos if c >= 500),
            "latencia_p50_ms": round(float(np.percentile(self.latencias_ms, 50)), 2) if hay else None,
            "latencia_p95_ms": round(float(np.percentile(self.latencias_ms, 95)), 2) if hay else None,
        }


metricas = Metricas()
monitor = None
if RUTA_REFERENCIA.exists():
    monitor = MonitorDeDeriva(json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8")))
app = FastAPI(title="GrantMatch: probabilidad de financiamiento")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Mide la latencia y el codigo de cada peticion a /predecir. Si la ruta
    lanza una excepcion, la cuenta como error de servidor (500)."""
    if request.url.path != "/predecir":
        return await call_next(request)
    inicio = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        metricas.registrar((time.perf_counter() - inicio) * 1000, 500)
        return JSONResponse(status_code=500, content={"detail": "Error interno al predecir"})
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)
    return respuesta


class Par(BaseModel):
    """Un award y una publicacion, con lo que se sabe al recomendar.

    Son obligatorios el campo de la publicacion y los paises de sus autores:
    del lado del investigador casi siempre estan (en la OBT, el campo falta
    en el 0.6 % de las publicaciones), y sin ellos las dos features saldrian
    0 sin ningun error. Del lado del award, el campo y el pais del
    financiador son opcionales: en OpenAlex el campo falta en el 97.5 % de
    los awards y el pais en el 4.7 %, y el modelo se entreno asi. El resto
    tambien puede llegar vacio: la preparacion lo imputa.
    """
    # Identificadores de OpenAlex (opcionales): no entran al modelo, pero
    # quedan en el registro para unir la prediccion con la verdad despues.
    award_key: str | None = None
    work_key: str | None = None
    # La convocatoria (award)
    award_field_name: str | None = None
    funder_country_code: str | None = Field(default=None, min_length=2, max_length=2)
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


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.get("/deriva")
def ver_deriva():
    """El PSI de la ventana actual contra la referencia, y si hay alerta."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="No hay referencia de deriva: corre python -m monitoreo.simular_deriva")
    return monitor.estado()


def registrar_prediccion(id_prediccion, par, probabilidad, recomendar):
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_REGISTRO, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps({
            "id_prediccion": id_prediccion,
            "momento": datetime.now().isoformat(timespec="seconds"),
            "version_modelo": version_modelo,
            "entrada": par.model_dump(mode="json"),
            "probabilidad_financiamiento": probabilidad,
            "recomendar": recomendar,
        }) + "\n")


@app.post("/predecir")
def predecir(par: Par):
    if modelo is None:
        raise HTTPException(status_code=503, detail="No hay modelo cargado: corre python -m servicio.exportar_modelo")
    pares = preparar_entrada(par)
    probabilidad = round(float(modelo.predict_proba(pares[NUMERICAS + CATEGORICAS + YA_LISTAS])[0, 1]), 4)
    recomendar = probabilidad >= UMBRAL
    if monitor is not None:
        monitor.registrar(pares.iloc[0].to_dict())
    id_prediccion = uuid.uuid4().hex
    registrar_prediccion(id_prediccion, par, probabilidad, recomendar)
    return {
        "id_prediccion": id_prediccion,
        "probabilidad_financiamiento": probabilidad,
        "recomendar": recomendar,
        "version_modelo": version_modelo,
    }
