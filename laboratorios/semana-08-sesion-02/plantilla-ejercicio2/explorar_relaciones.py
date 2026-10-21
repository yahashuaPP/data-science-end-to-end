import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
errores = (viajes["tarifa"] == 999.99) | (viajes["duracion_min"] < 0)
print(f"Errores confirmados: {errores.sum()}")
limpios = viajes[~errores].copy()
print(f"Filas que quedan: {len(limpios)} de {len(viajes)}")

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=8, alpha=0.4)
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.title("Distancia vs tarifa")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# Hay dos grupos fuera de la tendencia:
# a) una linea horizontal en tarifa = 70.
tarifa_fija = limpios[limpios["tarifa"] == 70]
print(f"\nViajes con tarifa exacta de 70: {len(tarifa_fija)}")
toca_jfk = (tarifa_fija["zona_origen"] == "JFK Airport") | (tarifa_fija["zona_destino"] == "JFK Airport")
print(f"De esos, con origen o destino JFK: {toca_jfk.sum()}")
print("Rango de distancia de esos viajes (km):",
      tarifa_fija["distancia_km"].min(), "a", tarifa_fija["distancia_km"].max())

# b) una linea vertical en distancia = 0 con tarifas de hasta ~30.
sin_distancia = limpios[limpios["distancia_km"] == 0]
print(f"\nViajes con distancia 0: {len(sin_distancia)}")
print("Tarifa de esos viajes:", sin_distancia["tarifa"].min(), "a", sin_distancia["tarifa"].max())

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
propinas = limpios.groupby("metodo_pago", dropna=False).agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    propina_cero=("propina", lambda p: (p == 0).sum()),
)
propinas["pct_propina_cero"] = 100 * propinas["propina_cero"] / propinas["viajes"]
print("\nPropina por metodo de pago:")
print(propinas.round(2))

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina"]
print("\nCorrelacion con la tarifa:")
print(limpios[columnas].corrwith(limpios["tarifa"]).round(3))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print(f"\nPasajeros vacio: {limpios['pasajeros'].isna().sum()}")
print(f"Pasajeros en 0: {(limpios['pasajeros'] == 0).sum()}")