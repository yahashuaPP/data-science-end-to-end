"""Entrena el modelo final de GrantMatch y lo registra como una version nueva
de "grantmatch" en el registro de modelos de MLflow.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.registrar_modelo           # 100 arboles (el modelo final)
    python -m servicio.registrar_modelo 200       # otro numero de arboles

Usa la misma division y la misma preparacion de modelado/pipeline_proyecto.py.
Cada vez que lo corres crea una version nueva, con su corrida y su AUC de
prueba: ninguna version reemplaza a otra. No mueve el alias "produccion":
eso lo hace servicio/promover.py, a mano, despues de mirar la metrica.
"""

import hashlib
import logging
import sys

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import (
    RUTA_DATOS, anio_de_corte, cargar_datos, cargar_y_dividir, construir_preparacion,
)

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "grantmatch"
EXPERIMENTO = "grantmatch-registro"


def huella(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()[:12]


if __name__ == "__main__":
    arboles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()

    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=f"gradient-boosting-{arboles}"):
        modelo = Pipeline([
            ("preparar", construir_preparacion()),
            ("modelo", GradientBoostingClassifier(n_estimators=arboles, random_state=42)),
        ])
        modelo.fit(X_ent, y_ent)
        auc = roc_auc_score(y_prueba, modelo.predict_proba(X_prueba)[:, 1])
        mlflow.log_params({
            "n_estimators": arboles,
            "division": "por fecha",
            "anio_de_corte": anio_de_corte(cargar_datos()),
            "huella_datos": huella(RUTA_DATOS),
        })
        mlflow.log_metric("auc_prueba", auc)
        info = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)

    print(f"Version {info.registered_model_version} de {NOMBRE}: gradient boosting de {arboles} arboles, AUC de prueba {auc:.4f}")
