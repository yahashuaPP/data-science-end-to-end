"""El desempeno del modelo de GrantMatch en produccion.

Une el registro de predicciones del servicio (monitoreo/predicciones.jsonl,
una linea por prediccion) con la verdad, y calcula por semana y por version
del modelo:

- Lo que se sabe el mismo dia (metricas indirectas): cuantas predicciones
  hubo, que fraccion se recomendo, la probabilidad promedio y que fraccion
  de los pares traia el campo del award (calidad de los datos).
- Lo que se sabe cuando llega la verdad: la AUC y el recall, la metrica de
  la Semana 11.

La verdad en GrantMatch tarda mucho: un par (award, publicacion) se confirma
cuando la publicacion aparece en OpenAlex reconociendo ese award, y eso pasa
meses o anos despues de la recomendacion. La verdad sale de exportar los
pares nuevos de GOLD.FACT_AWARD_WORKS con su fecha de carga
(monitoreo/verdad.csv: award_key, work_key, fecha_confirmacion). Una semana
solo entra al calculo cuando ya paso DIAS_MADUREZ desde su ultimo dia:
antes, un par sin confirmar todavia puede confirmarse.

Desde psets/proyecto/:

    python -m monitoreo.desempeno
    python -m monitoreo.desempeno 2027-12-31     # con otra fecha de hoy
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

CARPETA = Path(__file__).resolve().parent
RUTA_REGISTRO = CARPETA / "predicciones.jsonl"
RUTA_VERDAD = CARPETA / "verdad.csv"
DIAS_MADUREZ = 365
# Referencias del modelo aprobado (version 1, validacion por award, Semana 11).
AUC_REFERENCIA = 0.897
RECALL_REFERENCIA = 0.982


def cargar_registro(ruta=RUTA_REGISTRO):
    """Una fila por prediccion, con la entrada aplanada y la semana."""
    with open(ruta, encoding="utf-8") as archivo:
        filas = [json.loads(linea) for linea in archivo if linea.strip()]
    registro = pd.json_normalize(filas)
    registro.columns = [c.removeprefix("entrada.") for c in registro.columns]
    momento = pd.to_datetime(registro["momento"])
    registro["semana"] = momento.dt.to_period("W").dt.start_time.dt.date
    return registro


def metricas_inmediatas(registro):
    """Lo que se puede calcular el mismo dia, por semana y version."""
    return registro.groupby(["semana", "version_modelo"]).agg(
        predicciones=("id_prediccion", "size"),
        tasa_recomendacion=("recomendar", "mean"),
        probabilidad_media=("probabilidad_financiamiento", "mean"),
        con_campo_award=("award_field_name", lambda s: s.notna().mean()),
    ).round(3)


def desempeno_maduro(registro, verdad, hoy, dias_madurez=DIAS_MADUREZ):
    """AUC y recall de las semanas maduras, por semana y version. Un par es
    real si esta en la verdad con fecha_confirmacion hasta hoy."""
    hoy = pd.Timestamp(hoy)
    confirmados = verdad[pd.to_datetime(verdad["fecha_confirmacion"]) <= hoy]
    reales = set(zip(confirmados["award_key"], confirmados["work_key"]))
    pares = registro.dropna(subset=["award_key", "work_key"]).copy()
    domingo = pd.to_datetime(pares["semana"]) + pd.Timedelta(days=6)
    pares = pares[domingo + pd.Timedelta(days=dias_madurez) <= hoy]
    pares["real"] = [int((a, w) in reales) for a, w in zip(pares["award_key"], pares["work_key"])]

    filas = []
    for (semana, version), grupo in pares.groupby(["semana", "version_modelo"]):
        positivos = grupo[grupo["real"] == 1]
        filas.append({
            "semana": semana, "version_modelo": version, "pares": len(grupo), "reales": len(positivos),
            # La AUC necesita pares de las dos clases.
            "auc": round(roc_auc_score(grupo["real"], grupo["probabilidad_financiamiento"]), 3)
            if grupo["real"].nunique() == 2 else None,
            "recall": round(positivos["recomendar"].mean(), 3) if len(positivos) else None,
        })
    return pd.DataFrame(filas, columns=["semana", "version_modelo", "pares", "reales", "auc", "recall"])


if __name__ == "__main__":
    hoy = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    if not RUTA_REGISTRO.exists():
        sys.exit(f"No hay {RUTA_REGISTRO.name}: el servicio todavia no registro predicciones.")
    registro = cargar_registro()
    print(f"{len(registro)} predicciones registradas, del {registro['momento'].min()} al {registro['momento'].max()}")
    print("\nMetricas que se conocen el mismo dia (por semana y version):")
    print(metricas_inmediatas(registro).to_string())

    if not RUTA_VERDAD.exists():
        print(f"\nTodavia no hay {RUTA_VERDAD.name}: sin verdad no hay AUC ni recall.")
        sys.exit(0)
    tabla = desempeno_maduro(registro, pd.read_csv(RUTA_VERDAD), hoy)
    print(f"\nAUC y recall de las semanas maduras al {hoy} (referencia: AUC {AUC_REFERENCIA}, recall {RECALL_REFERENCIA}):")
    print(tabla.to_string(index=False) if len(tabla) else "Ninguna semana madura todavia.")
