import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# Muestra de la OBT de GrantMatch (OBT.OBT_AWARD_WORK), exportada desde
# Snowflake con exportar_obt.sql.
datos = pd.read_csv("obt_award_work_muestra.csv")
datos.columns = datos.columns.str.lower()  # Snowflake exporta en MAYUSCULAS

# El modelo predice si un par (award, publicacion) es pertinente.
OBJETIVO = "label"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))
print(f"Valores distintos de {OBJETIVO}:", datos[OBJETIVO].value_counts().to_dict())

# Paso 4: la senal que esperamos que mas influya en la pertinencia es que
# el award y la publicacion sean del mismo campo. Con solo positivos no
# se puede promediar el objetivo por grupo (siempre da 1), asi que se
# mira cuantos pares reales coinciden en campo.
datos["mismo_campo"] = datos["award_field_name"] == datos["work_field_name"]
datos["mismo_dominio"] = datos["award_domain_name"] == datos["work_domain_name"]
print("\nPares reales con el mismo campo:", round(100 * datos["mismo_campo"].mean(), 1), "%")
print("Pares reales con el mismo dominio:", round(100 * datos["mismo_dominio"].mean(), 1), "%")

por_campo = (datos.groupby("award_field_name")["mismo_campo"]
             .agg(pares="size", pct_mismo_campo="mean")
             .sort_values("pares", ascending=False).head(15))
por_campo["pct_mismo_campo"] = (100 * por_campo["pct_mismo_campo"]).round(1)
print(por_campo)

plt.figure(figsize=(8, 6))
plt.barh(por_campo.index[::-1], por_campo["pct_mismo_campo"][::-1])
plt.xlabel("% de pares reales con el mismo campo")
plt.title("Coincidencia de campo award-publicacion")
plt.tight_layout()
plt.savefig("figuras/coincidencia_campo.png")
plt.close()

# Paso 5: consultas para los 3 hallazgos.

# Hallazgo A: el objetivo es constante (solo hay positivos).
print("\n[A] Proporcion de label = 1:", datos[OBJETIVO].mean())

# Hallazgo B: concentracion. Pocos awards y funders acumulan muchos pares.
pares_por_award = datos["award_key"].value_counts()
top_1pct = pares_por_award.head(max(1, len(pares_por_award) // 100)).sum()
print(f"\n[B] Awards distintos: {len(pares_por_award)}")
print(f"[B] Pares del 1% de awards con mas publicaciones: {100 * top_1pct / len(datos):.1f}%")
print("[B] Pares por funder (top 10, %):")
print((100 * datos["funder_name"].value_counts(normalize=True).head(10)).round(1))

plt.figure()
plt.hist(datos["funded_outputs_count"].dropna(), bins=50, log=True)
plt.xlabel("Publicaciones financiadas por el award")
plt.ylabel("Pares (escala log)")
plt.title("Concentracion por award")
plt.savefig("figuras/concentracion_awards.png")
plt.close()

# Hallazgo C: texto disponible para los embeddings y montos.
print("\n[C] Pares sin descripcion del award (%):", round(100 * (1 - datos["has_description"].mean()), 1))
print("[C] Pares sin abstract (%):", round(100 * (1 - datos["has_abstract"].mean()), 1))
print("[C] Longitud del texto (caracteres):")
print(datos[["award_text_len", "work_text_len"]].describe().round(0))
print("[C] amount_usd vacio (%):", round(100 * datos["amount_usd"].isna().mean(), 1))
print("[C] Montos marcados invalidos:", int(datos["is_amount_invalid"].sum()))