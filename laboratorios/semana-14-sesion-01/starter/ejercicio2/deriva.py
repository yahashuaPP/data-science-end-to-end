"""El monitor de deriva de datos del servicio.

Guarda los ultimos `ventana` viajes que recibio el servicio y compara la
distribucion de distancia_km y de zona_origen con la de entrenamiento
(referencia_deriva.json), con el PSI del notebook del Ejercicio 1.

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK, y despues python simular_semanas.py.
"""

import numpy as np

MINIMO = 0.0001


def proporciones_numericas(valores, cortes):
    """Proporcion de los valores en cada casilla. `cortes` son los limites
    interiores: con 9 cortes hay 10 casillas, la primera hasta el primer
    corte y la ultima desde el ultimo, abiertas. Los None no cuentan."""
    # TODO 1: quita los None, cuenta con np.histogram usando como bins
    # [-np.inf] + cortes + [np.inf], y divide por el total. Si no queda
    # ningun valor, devuelve un arreglo de ceros del largo correcto.
    numeros = [v for v in valores if v is not None]
    if not numeros:
        return np.zeros(len(cortes) + 1)
    conteo, _ = np.histogram(numeros, bins=[-np.inf] + list(cortes) + [np.inf])
    return conteo / len(numeros)


def proporciones_categoricas(valores, categorias):
    """Proporcion de los valores en cada categoria, en el orden de
    `categorias`. Un valor que no esta en `categorias` no cuenta en ninguna
    casilla, pero si en el total."""
    # TODO 2: cuenta cuantos valores hay de cada categoria y divide por la
    # cantidad total de valores.
    if not valores:
        return np.zeros(len(categorias))
    return np.array([sum(1 for v in valores if v == c) for c in categorias]) / len(valores)


def psi(referencia, actual):
    """El PSI entre dos listas de proporciones, con MINIMO en las casillas
    vacias, como en el notebook."""
    # TODO 3: la misma funcion del notebook.
    r = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    a = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((a - r) * np.log(a / r)))


class MonitorDeDeriva:
    def __init__(self, referencia, ventana=300, umbral=0.15):
        self.referencia = referencia
        self.ventana = ventana
        self.umbral = umbral
        self.distancias = []
        self.zonas = []

    def registrar(self, viaje):
        """Agrega un viaje (un diccionario con distancia_km y zona_origen) y
        deja solo los ultimos `ventana`."""
        # Agrega la distancia (aunque sea None) y la zona del viaje, y si
        # alguna lista pasa de `ventana` elementos, quita los mas viejos.
        self.distancias.append(viaje.get("distancia_km"))
        self.zonas.append(viaje.get("zona_origen"))
        self.distancias = self.distancias[-self.ventana:]
        self.zonas = self.zonas[-self.ventana:]

    def estado(self):
        """Un diccionario con "viajes_en_ventana", "psi" (un diccionario con
        el PSI de distancia_km y de zona_origen, redondeado a 4 decimales) y
        "alerta": True solo si la ventana esta llena y alguno de los dos PSI
        es mayor que el umbral. Con la ventana vacia, "psi" es un diccionario
        vacio."""
        # Calcula los dos PSI contra self.referencia (sus "proporciones",
        # con los "cortes" o las "categorias" de cada feature) y arma el
        # diccionario.
        if not self.zonas:
            return {"viajes_en_ventana": 0, "psi": {}, "alerta": False}
        ref_d, ref_z = self.referencia["distancia_km"], self.referencia["zona_origen"]
        valores = {
            "distancia_km": round(psi(ref_d["proporciones"],
                                      proporciones_numericas(self.distancias, ref_d["cortes"])), 4),
            "zona_origen": round(psi(ref_z["proporciones"],
                                     proporciones_categoricas(self.zonas, ref_z["categorias"])), 4),
        }
        llena = len(self.zonas) >= self.ventana
        return {"viajes_en_ventana": len(self.zonas), "psi": valores,
                "alerta": bool(llena and max(valores.values()) > self.umbral)}
