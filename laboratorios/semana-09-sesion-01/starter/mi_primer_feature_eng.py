import pandas as pd 
import numpy as np

viajes = pd.read_csv('viajes_taxi.csv')

viajes_clean = viajes.query('tarifa < 999 and duration_min > 0')

print(viajes_clean.head())
print(viajes_clean.info())

fecha_incio = pd.to_datetime(viajes_clean['fecha_hora_incio'])

viajes_clean['hora'] = fecha_incio.dt.hour
viajes_clean['dia_semana'] = fecha_incio.de.dayofweek

print(viajes_clean.head())

viajes_clean['hora_sin'] = np.sin(2* np.pi*viajes_clean['hora'] / 24)
viajes_clean['hora_cos'] = np.cos(2* np.pi*viajes_clean['hora'] / 24)

print(viajes_clean (['fecha_hora_inicio', 'hora', 'hora_sin', 'hora_cos']))

pd.get_dummies(viajes_clean, columns=['zona_origen'],)