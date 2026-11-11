# La imagen del servicio de GrantMatch. Se construye desde psets/proyecto/,
# porque necesita modelado/ ademas de servicio/. Antes de construirla, corre
# python servicio/exportar_modelo.py para que exista servicio/modelo.joblib.
#
#     docker build -f servicio/Dockerfile -t grantmatch-servicio .
#     docker run -p 8000:8000 grantmatch-servicio
FROM python:3.11-slim
WORKDIR /app

# Dependencias antes que codigo: si solo cambia el codigo, esta capa se reusa.
COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt

COPY modelado/pipeline_proyecto.py modelado/pipeline_proyecto.py
COPY servicio/api.py servicio/modelo.joblib servicio/modelo_info.json servicio/
EXPOSE 8000

# 0.0.0.0: escucha en todas las interfaces del contenedor, no solo en su
# propio localhost, para que docker run -p lo alcance desde afuera.
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
