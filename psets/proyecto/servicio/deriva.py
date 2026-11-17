"""El monitor de deriva de datos del servicio de GrantMatch (Semana 14).

Guarda los ultimos `ventana` pares que recibio el servicio y compara, con el
PSI, la distribucion de tres features con la de entrenamiento:

- funder_en_pais_autor (0 o 1): la feature que mas pesa en el modelo;
- funder_country_code (los 10 paises mas frecuentes y "otro"): de donde
  vienen las convocatorias;
- cited_by_count (en los deciles de entrenamiento): el perfil del investigador.

La referencia (monitoreo/referencia_deriva.json) y el umbral salen de
monitoreo/simular_deriva.py.
"""

import numpy as np

MINIMO = 0.0001
OTRO = "otro"


def proporciones_numericas(valores, cortes):
    """Proporcion de los valores en cada casilla; `cortes` son los limites
    interiores (con 9 cortes, 10 casillas abiertas en los extremos)."""
    numeros = [v for v in valores if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not numeros:
        return np.zeros(len(cortes) + 1)
    conteo, _ = np.histogram(numeros, bins=[-np.inf] + list(cortes) + [np.inf])
    return conteo / len(numeros)


def proporciones_categoricas(valores, categorias):
    """Proporcion de cada categoria, en el orden de `categorias`. Si
    `categorias` incluye OTRO, ahi caen los valores que no estan en la lista."""
    if not valores:
        return np.zeros(len(categorias))
    conocidas = set(categorias)
    if OTRO in conocidas:
        valores = [v if v in conocidas else OTRO for v in valores]
    return np.array([sum(1 for v in valores if v == c) for c in categorias]) / len(valores)


def psi(referencia, actual):
    r = np.clip(np.asarray(referencia, dtype=float), MINIMO, None)
    a = np.clip(np.asarray(actual, dtype=float), MINIMO, None)
    return float(np.sum((a - r) * np.log(a / r)))


def construir_referencia(pares):
    """La referencia de deriva a partir de los pares de entrenamiento (un
    DataFrame con las features ya calculadas)."""
    cortes = np.unique(np.quantile(pares["cited_by_count"].dropna(), np.linspace(0.1, 0.9, 9))).tolist()
    paises = pares["funder_country_code"].fillna(OTRO).value_counts().head(10).index.tolist()
    paises = [p for p in paises if p != OTRO] + [OTRO]
    return {
        "funder_en_pais_autor": {"categorias": [0, 1],
                                 "proporciones": proporciones_categoricas(pares["funder_en_pais_autor"].tolist(), [0, 1]).tolist()},
        "funder_country_code": {"categorias": paises,
                                "proporciones": proporciones_categoricas(pares["funder_country_code"].fillna(OTRO).tolist(), paises).tolist()},
        "cited_by_count": {"cortes": cortes,
                           "proporciones": proporciones_numericas(pares["cited_by_count"].tolist(), cortes).tolist()},
    }


def psi_de_ventana(referencia, pares):
    """El PSI de cada feature de una ventana de pares (lista de diccionarios)."""
    columna = lambda nombre: [p.get(nombre) for p in pares]
    paises = [p.get("funder_country_code") or OTRO for p in pares]
    return {
        "funder_en_pais_autor": round(psi(referencia["funder_en_pais_autor"]["proporciones"],
                                          proporciones_categoricas(columna("funder_en_pais_autor"), [0, 1])), 4),
        "funder_country_code": round(psi(referencia["funder_country_code"]["proporciones"],
                                         proporciones_categoricas(paises, referencia["funder_country_code"]["categorias"])), 4),
        "cited_by_count": round(psi(referencia["cited_by_count"]["proporciones"],
                                    proporciones_numericas(columna("cited_by_count"), referencia["cited_by_count"]["cortes"])), 4),
    }


class MonitorDeDeriva:
    def __init__(self, referencia, ventana=500, umbral=0.10):
        self.referencia = referencia
        self.ventana = ventana
        self.umbral = umbral
        self.pares = []

    def registrar(self, par):
        """Agrega un par (un diccionario con las tres features) y deja solo
        los ultimos `ventana`."""
        self.pares.append({
            "funder_en_pais_autor": int(par.get("funder_en_pais_autor") or 0),
            "funder_country_code": par.get("funder_country_code"),
            "cited_by_count": par.get("cited_by_count"),
        })
        self.pares = self.pares[-self.ventana:]

    def estado(self):
        if not self.pares:
            return {"pares_en_ventana": 0, "psi": {}, "umbral": self.umbral, "alerta": False}
        valores = psi_de_ventana(self.referencia, self.pares)
        llena = len(self.pares) >= self.ventana
        return {"pares_en_ventana": len(self.pares), "psi": valores, "umbral": self.umbral,
                "alerta": bool(llena and max(valores.values()) > self.umbral)}
