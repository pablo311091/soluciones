# Usar una imagen oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt requirements.txt

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# --- Paso Crítico ---
# Descargar los corpus de sentimiento que TextBlob necesita
RUN python -m textblob.download_corpora

# Copiar el código de la aplicación
COPY sentiment_webhook.py .

# Comando para ejecutar la aplicación con Gunicorn (servidor de producción)
# Gunicorn usará la variable $PORT que Railway le proporciona automáticamente.
CMD ["gunicorn", "sentiment_webhook:app", "--bind", "0.0.0.0:$PORT"]
