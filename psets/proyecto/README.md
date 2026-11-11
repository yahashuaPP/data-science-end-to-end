# [Nombre de tu proyecto]

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

[Un párrafo: qué problema de negocio resuelve, para quién, y qué predice el modelo. Sale de tu propuesta del PSet 1, que está completa en `docs/`.]

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

[Qué notebook corre cada etapa del modelado, en qué orden, y qué necesita levantado (MLflow). Se completa con el PSet 3.]

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

El modelo se sirve por lotes cada noche (la razón está en [`docs/despliegue.md`](docs/despliegue.md)); `servicio/` es la API con que otro sistema le pide una predicción suelta al mismo modelo. Todo se corre desde `psets/proyecto/`.

**1. Exportar el modelo final desde MLflow.** Con el servidor de MLflow levantado (puerto 5050), el script busca en el experimento `grantmatch` la corrida marcada con `candidato` = `final` y la guarda como `servicio/modelo.joblib`, junto con `servicio/modelo_info.json` (su `run_id`). Ninguno de los dos se sube al repositorio.

```
python servicio/exportar_modelo.py            # la corrida final
python servicio/exportar_modelo.py <run_id>   # otra corrida, por ejemplo para volver a la anterior
```

**2. Probarlo sin contenedor.**

```
uvicorn servicio.api:app --port 8000
```

La documentación interactiva queda en el puerto 8000, ruta `/docs`. Las pruebas (`pytest psets/proyecto -v`, desde la raíz del fork) no necesitan el modelo exportado.

**3. Construir y levantar el contenedor.** La imagen se construye desde `psets/proyecto/` porque necesita `modelado/pipeline_proyecto.py` además de `servicio/`:

```
docker build -f servicio/Dockerfile -t grantmatch-servicio .
docker run -p 8000:8000 grantmatch-servicio
```

**4. Pedir una predicción.** Se manda el award y el perfil del investigador como se conocen. El servicio calcula `mismo_campo` y `funder_en_pais_autor` con `agregar_features` de `modelado/pipeline_proyecto.py`:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"award_field_name": "Physics and Astronomy", "funder_country_code": "US", "amount_usd": 250000, "duration_days": 1095, "funding_type": "grant", "award_domain_name": "Physical Sciences", "is_global_south": false, "work_field_name": "Physics and Astronomy", "author_country_codes": ["US", "DE"], "work_domain_name": "Physical Sciences", "authors_count": 4, "cited_by_count": 12}'
```

Respuesta (la probabilidad depende del modelo exportado):

```
{"probabilidad_financiamiento": ⟨X⟩, "recomendar": ⟨true/false⟩, "version_modelo": "⟨run_id⟩"}
```

`recomendar` es `true` cuando la probabilidad alcanza el umbral de menor costo de la validación (variable de entorno `UMBRAL`, 0.10 por defecto). Un par sin `award_field_name`, `funder_country_code`, `work_field_name` o `author_country_codes` se rechaza con 422: sin ellos las dos features saldrían 0 sin dar error.
