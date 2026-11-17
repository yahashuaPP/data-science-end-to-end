"""Exporta la version de "grantmatch" que tiene el alias "produccion" a
servicio/modelo.joblib, y su numero a servicio/modelo_version.txt.

La imagen del servicio lleva los dos archivos adentro: el contenedor no
necesita llegar al servidor de MLflow para saber que modelo sirve. Ninguno
de los dos se sube al repositorio: se regeneran con este script.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.exportar_modelo
"""

import logging
from pathlib import Path

import joblib
import mlflow.sklearn
from mlflow import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo.joblib"
RUTA_VERSION = CARPETA / "modelo_version.txt"
NOMBRE = "grantmatch"
ALIAS = "produccion"

if __name__ == "__main__":
    version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS).version
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version}")
    joblib.dump(modelo, RUTA_MODELO)
    RUTA_VERSION.write_text(version, encoding="utf-8")
    print(f"{NOMBRE}@{ALIAS} = version {version}, guardada en {RUTA_MODELO.name} y {RUTA_VERSION.name}")
