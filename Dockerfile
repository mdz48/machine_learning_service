FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para paquetes de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Modelo de spaCy para la etapa de negacion + tokenizacion de zonas del NLP.
# NO viene en requirements.txt; sin el, el pipeline NLP falla al cargar (spacy.load).
RUN python -m spacy download es_core_news_sm

# Copiar todo el codigo fuente
COPY . .

# Exponer el puerto
EXPOSE 8001

# Comando para iniciar el servicio (apunta a app.main:app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
