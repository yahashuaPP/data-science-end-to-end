"""Pruebas de monitoreo/desempeno.py con un registro y una verdad de pocas
filas, escritos a mano. Desde la raiz del fork: pytest psets/proyecto -v"""

import json

import pandas as pd

from monitoreo.desempeno import cargar_registro, desempeno_maduro, metricas_inmediatas


def escribir_registro(ruta):
    filas = [
        # Semana del 2 de noviembre de 2026, version 1: dos pares reales, dos no.
        ("a", "2026-11-02T10:00:00", "1", "G1", "W1", 0.90, True),
        ("b", "2026-11-03T10:00:00", "1", "G2", "W2", 0.80, True),
        ("c", "2026-11-04T10:00:00", "1", "G3", "W3", 0.30, True),
        ("d", "2026-11-05T10:00:00", "1", "G4", "W4", 0.05, False),
        # Semana del 9, version 2: todavia no madura en la prueba.
        ("e", "2026-11-10T10:00:00", "2", "G5", "W5", 0.70, True),
    ]
    with open(ruta, "w") as archivo:
        for i, momento, version, award, work, prob, recomendar in filas:
            archivo.write(json.dumps({
                "id_prediccion": i, "momento": momento, "version_modelo": version,
                "entrada": {"award_key": award, "work_key": work, "award_field_name": None},
                "probabilidad_financiamiento": prob, "recomendar": recomendar,
            }) + "\n")


def test_metricas_inmediatas_por_semana_y_version(tmp_path):
    escribir_registro(tmp_path / "predicciones.jsonl")
    tabla = metricas_inmediatas(cargar_registro(tmp_path / "predicciones.jsonl"))
    assert list(tabla["predicciones"]) == [4, 1]
    assert list(tabla["tasa_recomendacion"]) == [0.75, 1.0]


def test_solo_entran_las_semanas_maduras_y_la_verdad_hasta_hoy(tmp_path):
    escribir_registro(tmp_path / "predicciones.jsonl")
    registro = cargar_registro(tmp_path / "predicciones.jsonl")
    verdad = pd.DataFrame({
        "award_key": ["G1", "G3", "G5"], "work_key": ["W1", "W3", "W5"],
        "fecha_confirmacion": ["2027-03-01", "2027-06-01", "2028-01-01"],
    })
    # Al 14 de noviembre de 2027, con 365 dias de madurez: solo la semana del 2
    # (domingo 8 + 365 = 8 de noviembre de 2027). G5 se confirma despues.
    tabla = desempeno_maduro(registro, verdad, "2027-11-14")
    assert len(tabla) == 1
    fila = tabla.iloc[0]
    assert (fila["pares"], fila["reales"]) == (4, 2)
    assert fila["recall"] == 1.0           # los dos reales se recomendaron
    assert fila["auc"] == 0.75             # G3 (real, 0.30) queda debajo de G2 (no real, 0.80)
