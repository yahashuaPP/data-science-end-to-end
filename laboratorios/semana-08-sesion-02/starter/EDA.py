import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('figuras', exist_ok=True)

data = pd.read_csv('./starter/viajes_taxi.csv')

print('EDA')

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicate().sum())

print(data.describe())

plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title('Distribucion de la tarifa')
plt.xlabel('USD')
plt.ylabel('N')
plt.savefig('./starter/figuras/hist.png')
plt.show()