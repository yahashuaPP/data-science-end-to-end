import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('figuras', exist_ok=True)

data = pd.read_csv('./viajes_taxi.csv')

print('EDA')

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())

print(data.describe())

plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title('Distribucion de la tarifa')
plt.xlabel('USD')
plt.ylabel('N')
plt.savefig('./figuras/hist.png')
plt.close()

plt.figure()
plt.boxplot(data['tarifa'])
plt.title('Boxplot de la tarifa')
plt.savefig('./figuras/boxplot.png')
plt.close()

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)

iqr = q3 - q1

ls = q3 + 1.5 * iqr
li = q1 - 1.5 * iqr

atipicos = data.query(f'tarifa < {li} or tarifa > {ls}')
sin_atipicos = data.query(f'tarifa > {li} and tarifa < {ls}')

print(atipicos)

plt.figure()
plt.scatter(data['tarifa'], data['distancia_km'])
plt.title('Dispersion de la tarifa vs la distancia')
plt.savefig('./figuras/disp_tarifa_vs_distancia.png')
plt.close()

print(data.corr(numeric_only=True))

plt.figure()
plt.scatter(sin_atipicos['tarifa'], sin_atipicos['distancia_km'])
plt.title('Dispersion de la tarifa vs la distancia sin atipicos')
plt.savefig('./figuras/disp_tarifa_vs_distancia_no_atipicos.png')
plt.close()