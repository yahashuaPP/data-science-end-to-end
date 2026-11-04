"""Ejercicio 3 (Semana 10, Sesion 1): los primeros modelos de GrantMatch.

Usa la division y la preparacion de
semana-09-sesion-02/plantilla-pipeline/pipeline_proyecto.py. Es
clasificacion: el par (award, publicacion) es un financiamiento real o no.
Metrica: area bajo la curva ROC (AUC), que no depende de un umbral; se
reporta tambien la exactitud.

    python3 modelos_proyecto.py

Escribe resultados_proyecto.csv en esta carpeta.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent.parent / "semana-09-sesion-02" / "plantilla-pipeline"))
from pipeline_proyecto import cargar_datos, dividir, preparacion  # noqa: E402

datos = cargar_datos()
X_ent, X_prueba, y_ent, y_prueba, corte = dividir(datos)
print(f"Entrenamiento: {len(X_ent)} pares (awards antes de {corte})   prueba: {len(X_prueba)}")

resultados = []


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    prob_ent = pipeline.predict_proba(X_ent)[:, 1]
    prob_prueba = pipeline.predict_proba(X_prueba)[:, 1]
    auc_ent = roc_auc_score(y_ent, prob_ent)
    auc_prueba = roc_auc_score(y_prueba, prob_prueba)
    resultados.append({
        "modelo": nombre,
        "auc_entrenamiento": round(auc_ent, 4),
        "auc_prueba": round(auc_prueba, 4),
        "brecha": round(auc_ent - auc_prueba, 4),
        "exactitud_prueba": round(accuracy_score(y_prueba, pipeline.predict(X_prueba)), 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:34s} AUC entrenamiento {auc_ent:.4f}   prueba {auc_prueba:.4f}   {segundos:6.2f} s")
    return pipeline


# Linea base: predice siempre la proporcion de pares reales del
# entrenamiento. Con clases balanceadas su AUC es 0.5: es lo que saca
# cualquier modelo que no distinga un par real de uno al azar.
evaluar("linea base (DummyClassifier)", DummyClassifier(strategy="prior"))

# Modelo lineal, con la regularizacion por defecto (C=1) y una mas fuerte.
evaluar("logistica C=1", LogisticRegression(C=1.0, max_iter=1000))
evaluar("logistica C=0.01", LogisticRegression(C=0.01, max_iter=1000))

# Ensamblajes: un bosque sin limite, el mismo con profundidad limitada,
# y gradient boosting.
evaluar("bosque sin limite", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
evaluar("bosque max_depth=8", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
evaluar("gradient boosting", GradientBoostingClassifier(random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv(CARPETA / "resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
mejor = tabla.loc[tabla["auc_prueba"].idxmax()]
base = tabla.loc[0, "auc_prueba"]
print()
print(f"Mejor en prueba: {mejor['modelo']} (AUC {mejor['auc_prueba']:.4f}, "
      f"{mejor['auc_prueba'] - base:+.4f} sobre la linea base)")
