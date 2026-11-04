"""Ejercicio 3 (Semana 10, Sesion 2): ensamblaje, red neuronal y LLM
para GrantMatch.

Primera corrida: entrena los dos modelos y escribe prompt_llm_proyecto.txt
(20 pares de entrenamiento con su label como ejemplos y 10 pares de prueba
sin label). Copialo completo en un LLM de chat y guarda sus 10 respuestas
(0 o 1, una por linea) en respuestas_llm_proyecto.txt, en esta carpeta.
Despues vuelve a correrlo:

    python3 comparar_proyecto.py

Escribe comparacion_proyecto.csv en esta carpeta.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

CARPETA = Path(__file__).resolve().parent
LABS = CARPETA.parent.parent
sys.path.insert(0, str(LABS / "semana-09-sesion-02" / "plantilla-pipeline"))
from pipeline_proyecto import cargar_datos, dividir, preparacion  # noqa: E402

RUTA_RESULTADOS_ANTERIORES = LABS / "semana-10-sesion-01" / "proyecto" / "resultados_proyecto.csv"
RUTA_PROMPT = CARPETA / "prompt_llm_proyecto.txt"
RUTA_RESPUESTAS = CARPETA / "respuestas_llm_proyecto.txt"

# Precio por millon de tokens de entrada del LLM que uses. 4.0 es Claude
# Opus 5.5 en la API de Anthropic (nov-2026); cambialo si usas otro.
PRECIO_POR_MILLON_DE_TOKENS = 4.0

ENSAMBLAJES = {
    "bosque sin limite": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "bosque max_depth=8": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "gradient boosting": lambda: GradientBoostingClassifier(random_state=42),
}

datos = cargar_datos()
X_ent, X_prueba, y_ent, y_prueba, corte = dividir(datos)

# El mejor ensamblaje de la sesion pasada, segun resultados_proyecto.csv.
anteriores = pd.read_csv(RUTA_RESULTADOS_ANTERIORES)
anteriores = anteriores[anteriores["modelo"].isin(ENSAMBLAJES)]
nombre_ensamblaje = anteriores.loc[anteriores["auc_prueba"].idxmax(), "modelo"]
print("Mejor ensamblaje de la Semana 10.1:", nombre_ensamblaje)

resultados = []


def entrenar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    auc = roc_auc_score(y_prueba, pipeline.predict_proba(X_prueba)[:, 1])
    fila = {"opcion": nombre, "auc_prueba": round(auc, 4), "segundos": round(segundos, 3), "parametros": None}
    resultados.append(fila)
    print(f"{nombre:22s} AUC prueba {auc:.4f}   {segundos:6.2f} s")
    return pipeline, fila


ensamblaje, _ = entrenar(nombre_ensamblaje, ENSAMBLAJES[nombre_ensamblaje]())
red, fila_red = entrenar("red neuronal", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42))
mlp = red.named_steps["modelo"]
fila_red["parametros"] = sum(m.size for m in mlp.coefs_) + sum(v.size for v in mlp.intercepts_)
print("Parametros de la red:", fila_red["parametros"])

# La prueba del LLM: 10 pares de prueba (5 reales, 5 al azar) y 20 de
# entrenamiento como ejemplos (10 y 10), en columnas legibles.
LEGIBLES = [
    "funder_name", "funder_country_code", "funding_type", "award_field_name", "amount_usd",
    "work_field_name", "author_country_codes", "authors_count",
]
prueba_completa = datos.loc[X_prueba.index]
entrenamiento_completo = datos.loc[X_ent.index]
muestra = pd.concat([
    prueba_completa[prueba_completa["label"] == 1].sample(5, random_state=7),
    prueba_completa[prueba_completa["label"] == 0].sample(5, random_state=7),
]).sample(frac=1, random_state=7)
ejemplos = pd.concat([
    entrenamiento_completo[entrenamiento_completo["label"] == 1].sample(10, random_state=7),
    entrenamiento_completo[entrenamiento_completo["label"] == 0].sample(10, random_state=7),
]).sample(frac=1, random_state=7)

prompt = (
    "Cada fila es un par entre una subvencion (award) y una publicacion cientifica. "
    "label = 1 si la subvencion financio esa publicacion, 0 si no. Abajo hay 20 pares con su label "
    "y 10 pares sin label. Responde solo con 10 numeros (0 o 1), uno por linea, en el mismo orden "
    "de los pares sin label, sin texto adicional. amount_usd esta en dolares; "
    "author_country_codes son los paises de los autores separados por |.\n\n"
    "Pares con label:\n"
    + ejemplos[LEGIBLES + ["label"]].round({"amount_usd": 0}).to_csv(index=False)
    + "\nPares sin label:\n"
    + muestra[LEGIBLES].round({"amount_usd": 0}).to_csv(index=False)
)
RUTA_PROMPT.write_text(prompt, encoding="utf-8")
tokens_por_prompt = len(prompt) / 4
print()
print(f"{RUTA_PROMPT.name} escrito: {len(prompt)} caracteres (~{tokens_por_prompt:.0f} tokens)")

y_muestra = muestra["label"]
X_muestra = X_prueba.loc[muestra.index]
exactitud = {
    nombre_ensamblaje: accuracy_score(y_muestra, ensamblaje.predict(X_muestra)),
    "red neuronal": accuracy_score(y_muestra, red.predict(X_muestra)),
}

if RUTA_RESPUESTAS.exists():
    respuestas = [int(float(l)) for l in RUTA_RESPUESTAS.read_text().split() if l.strip()]
    if len(respuestas) != 10:
        raise SystemExit(f"{RUTA_RESPUESTAS.name} debe tener 10 numeros, tiene {len(respuestas)}")
    exactitud["LLM"] = accuracy_score(y_muestra, respuestas)
    # Costo de evaluar 10 000 pares: un prompt como este cada 10 pares.
    costo_10000 = (10_000 / 10) * tokens_por_prompt * PRECIO_POR_MILLON_DE_TOKENS / 1_000_000
    resultados.append({"opcion": "LLM", "auc_prueba": None, "segundos": 0.0, "parametros": None})
else:
    costo_10000 = None
    print(f"Falta {RUTA_RESPUESTAS.name}: pega el prompt en un LLM y guarda sus 10 respuestas.")

tabla = pd.DataFrame(resultados)
tabla["exactitud_10_pares"] = tabla["opcion"].map(exactitud).round(2)
tabla["costo_10000_pares_usd"] = [0.0, 0.0] + ([round(costo_10000, 2)] if costo_10000 is not None else [])
tabla["explicable"] = ["importancia de columnas", "no (pesos sin significado)"] + (
    ["texto, sin garantia de fidelidad"] if costo_10000 is not None else [])
tabla.to_csv(CARPETA / "comparacion_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))

importancias = pd.Series(
    ensamblaje.named_steps["modelo"].feature_importances_,
    index=ensamblaje.named_steps["preparar"].get_feature_names_out(),
).sort_values(ascending=False)
print()
print("Las 5 columnas mas importantes para el ensamblaje:")
print(importancias.head(5).round(4).to_string())
