"""Exporta el modelo final de GrantMatch desde MLflow a servicio/modelo.joblib.

Busca en el experimento "grantmatch" la corrida marcada con candidato = final
(la marca experimentos_proyecto.ipynb), carga su pipeline y lo guarda con
joblib. Junto al modelo escribe modelo_info.json con el run_id, para que el
servicio diga en cada respuesta que modelo la produjo. Ninguno de los dos
archivos se sube: se regeneran con este script.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py
    python servicio/exportar_modelo.py <run_id>    # otra corrida, p. ej. para volver a la anterior
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo.joblib"
RUTA_INFO = CARPETA / "modelo_info.json"
EXPERIMENTO = "grantmatch"


def run_id_final():
    experimento = mlflow.get_experiment_by_name(EXPERIMENTO)
    if experimento is None:
        raise SystemExit(f"No existe el experimento {EXPERIMENTO!r} en {mlflow.get_tracking_uri()}")
    corridas = mlflow.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="tags.candidato = 'final'",
        order_by=["attributes.start_time DESC"],
    )
    if corridas.empty:
        raise SystemExit("Ninguna corrida tiene candidato = final: corre experimentos_proyecto.ipynb")
    return corridas.loc[0, "run_id"]


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else run_id_final()
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    joblib.dump(modelo, RUTA_MODELO)

    corrida = mlflow.get_run(run_id)
    info = {
        "run_id": run_id,
        "nombre": corrida.data.tags.get("mlflow.runName"),
        "auc_prueba": corrida.data.metrics.get("auc_prueba"),
        "exportado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RUTA_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Modelo {info['nombre']} ({run_id}) guardado en {RUTA_MODELO.name}")
