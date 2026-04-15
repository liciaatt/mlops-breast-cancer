# On part d'une image Python officielle légère
FROM python:3.11-slim

# On définit le dossier de travail dans le conteneur
WORKDIR /app

# On copie et installe les dépendances en premier
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie les fichiers du projet
COPY train.py .
COPY app.py .

# On entraîne le modèle pendant le build
RUN python train.py

# On expose le port 8000
EXPOSE 8000

# On démarre l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]