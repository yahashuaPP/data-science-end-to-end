"""Pone el alias "produccion" en una version de "grantmatch".

Desde psets/proyecto/:

    python -m servicio.promover 2

Muestra la AUC de prueba de la version nueva y la de la que estaba en
produccion, y deja en la version una etiqueta con quien la promovio y contra
que version se comparo. Para volver al modelo anterior basta con promover
su numero otra vez.
"""

import getpass
import logging
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "grantmatch"
ALIAS = "produccion"


def auc_de(cliente, version):
    run_id = cliente.get_model_version(NOMBRE, version).run_id
    return cliente.get_run(run_id).data.metrics.get("auc_prueba")


if __name__ == "__main__":
    nueva = sys.argv[1]
    cliente = MlflowClient()
    try:
        anterior = cliente.get_model_version_by_alias(NOMBRE, ALIAS).version
    except MlflowException:
        anterior = None

    cliente.set_registered_model_alias(NOMBRE, ALIAS, nueva)
    cliente.set_model_version_tag(NOMBRE, nueva, "promovida_por", getpass.getuser())
    cliente.set_model_version_tag(NOMBRE, nueva, "reemplaza_a", anterior or "ninguna")

    print(f"{NOMBRE}@{ALIAS}: version {nueva} (AUC de prueba {auc_de(cliente, nueva):.4f})")
    if anterior is not None and anterior != nueva:
        print(f"Antes estaba la version {anterior} (AUC de prueba {auc_de(cliente, anterior):.4f})."
              f" Para volver a ella: python -m servicio.promover {anterior}")
