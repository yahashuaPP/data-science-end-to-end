"""La deriva de GrantMatch, simulada con los pares de prueba (Semana 14).

Entrena el modelo final (el gradient boosting de 100 arboles, como la
version 1 del registro), toma ventanas de 500 pares de prueba (lo que el
servicio puntua en una noche de recomendaciones) y mide en cada una el PSI
de las tres features que vigila el servicio y la AUC del modelo:

- ventanas normales: cuanto se mueve el PSI sin que nada cambie;
- deriva de datos: una temporada con muchas mas convocatorias de China
  (los pares con funder_country_code = CN salen `peso` veces mas);
- deriva conceptual: los financiadores empiezan a financiar a autores de
  otros paises (programas internacionales). Los pares son los mismos; en una
  fraccion de los reales con funder_en_pais_autor = 1 el financiamiento ya
  no ocurre, y en la misma cantidad de pares con 0, si.

Elige el umbral de la alerta y guarda la referencia que usa el servicio en
monitoreo/referencia_deriva.json. Desde psets/proyecto/:

    python -m monitoreo.simular_deriva
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import (
    CATEGORICAS, NUMERICAS, OBJETIVO, YA_LISTAS, cargar_datos, cargar_y_dividir, construir_preparacion,
)
from servicio.deriva import construir_referencia, psi_de_ventana

CARPETA = Path(__file__).resolve().parent
RUTA_REFERENCIA = CARPETA / "referencia_deriva.json"
VENTANA = 500
COLUMNAS = NUMERICAS + CATEGORICAS + YA_LISTAS
VIGILADAS = ["funder_en_pais_autor", "funder_country_code", "cited_by_count"]

datos = cargar_datos()
X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
modelo = Pipeline([
    ("preparar", construir_preparacion()),
    ("modelo", GradientBoostingClassifier(random_state=42)),
]).fit(X_ent, y_ent)
prueba = datos.loc[X_prueba.index].copy()
prueba["probabilidad"] = modelo.predict_proba(prueba[COLUMNAS])[:, 1]
# La referencia es la prueba (awards desde el ano de corte), no el
# entrenamiento: es el periodo mas reciente, el mas parecido a lo que el
# servicio recibe, y en el que se valido el modelo (AUC 0.90). Contra el
# entrenamiento, la prueba entera ya tiene deriva (ver abajo).
referencia_entrenamiento = construir_referencia(datos.loc[X_ent.index])
referencia = construir_referencia(prueba)


def medir(ventana):
    valores = psi_de_ventana(referencia, ventana[VIGILADAS].to_dict("records"))
    return {**valores, "auc": round(roc_auc_score(ventana[OBJETIVO], ventana["probabilidad"]), 4)}


def ventana_normal(semilla):
    return prueba.sample(VENTANA, replace=True, random_state=semilla)


def ventana_china(semilla, peso):
    pesos = np.where(prueba["funder_country_code"] == "CN", peso, 1)
    return prueba.sample(VENTANA, replace=True, weights=pesos, random_state=semilla)


def ventana_internacional(semilla, fraccion):
    ventana = ventana_normal(semilla).copy()
    generador = np.random.default_rng(semilla)
    reales_locales = ventana.index[(ventana[OBJETIVO] == 1) & (ventana["funder_en_pais_autor"] == 1)]
    negativos_externos = ventana.index[(ventana[OBJETIVO] == 0) & (ventana["funder_en_pais_autor"] == 0)]
    n = min(int(fraccion * len(reales_locales)), len(negativos_externos))
    posiciones = np.arange(len(ventana))
    reales = posiciones[(ventana[OBJETIVO] == 1).to_numpy() & (ventana["funder_en_pais_autor"] == 1).to_numpy()]
    externos = posiciones[(ventana[OBJETIVO] == 0).to_numpy() & (ventana["funder_en_pais_autor"] == 0).to_numpy()]
    columna = ventana.columns.get_loc(OBJETIVO)
    ventana.iloc[generador.choice(reales, n, replace=False), columna] = 0
    ventana.iloc[generador.choice(externos, n, replace=False), columna] = 1
    return ventana


if __name__ == "__main__":
    print(f"AUC en toda la prueba: {roc_auc_score(prueba[OBJETIVO], prueba['probabilidad']):.4f}")
    print("PSI de toda la prueba (awards desde el corte) contra el entrenamiento:",
          psi_de_ventana(referencia_entrenamiento, prueba[VIGILADAS].to_dict("records")))
    print("Proporcion de pares con financiador de China: entrenamiento",
          round((datos.loc[X_ent.index, "funder_country_code"] == "CN").mean(), 3),
          " prueba", round((prueba["funder_country_code"] == "CN").mean(), 3))
    normales = pd.DataFrame([medir(ventana_normal(s)) for s in range(200)])
    print("\n200 ventanas normales de", VENTANA, "pares:")
    print(normales.describe(percentiles=[0.95]).loc[["mean", "95%", "max"]].round(4).to_string())

    filas = []
    for peso in [1, 2, 4, 8]:
        medidas = pd.DataFrame([medir(ventana_china(s, peso)) for s in range(20)]).mean()
        filas.append({"escenario": f"datos: China x{peso}", **medidas.round(4).to_dict()})
    for fraccion in [0.0, 0.25, 0.5, 0.75]:
        medidas = pd.DataFrame([medir(ventana_internacional(s, fraccion)) for s in range(20)]).mean()
        filas.append({"escenario": f"conceptual: {int(fraccion * 100)}% internacional", **medidas.round(4).to_dict()})
    print("\nDerivas simuladas (promedio de 20 ventanas):")
    print(pd.DataFrame(filas).to_string(index=False))

    chinas = pd.DataFrame([medir(ventana_china(s, 4)) for s in range(200)])
    alerta = lambda m, u: (m[VIGILADAS] > u).any(axis=1)
    tabla = pd.DataFrame([
        {"umbral": u, "falsas_alarmas": alerta(normales, u).mean(), "deteccion_china_x4": alerta(chinas, u).mean()}
        for u in [0.05, 0.10, 0.15, 0.20]
    ])
    print("\nUmbral de la alerta (200 ventanas normales y 200 con China x4):")
    print(tabla.to_string(index=False))

    RUTA_REFERENCIA.write_text(json.dumps(referencia, indent=2), encoding="utf-8")
    print(f"\nReferencia guardada en {RUTA_REFERENCIA.name}")
