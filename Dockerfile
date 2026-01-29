# Dockerfile simple pour l'API FastAPI
FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Copie requirements et installe dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code + le fichier des colonnes attendues
COPY deployment/api/ ./deployment/api/
COPY expected_columns.joblib ./deployment/api/

# Expose le port
EXPOSE 8000

# Lance l'API
CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]