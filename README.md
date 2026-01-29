# House Prices Prediction - Projet MLOps End-to-End

## Problème métier
Prédire le prix de vente de maisons à Ames (Iowa) à partir de 79 features (taille, qualité, année, garage, etc.).  
Dataset : Kaggle House Prices - Advanced Regression Techniques (~1460 lignes).

## Technologies utilisées
- Tracking & Registry : MLflow
- Modèle : RandomForestRegressor (scikit-learn)
- API : FastAPI + Uvicorn
- Préprocessing : pandas + get_dummies

## Comment lancer le projet (local)

1. Cloner le repo
git clone <ton-repo>

2. Créer et activer l'environnement
python -m venv mlops_env
.\mlops_env\Scripts\Activate.ps1   # Windows

3. Installer les dépendances
pip install -r requirements.txt

4. Lancer MLflow UI (terminal 1)
mlflow ui

5. Entraîner le modèle (terminal 2)
python -m src.models.train

6. Lancer l'API de prédiction (terminal 3)
uvicorn deployment.api.app:app --reload --port 8000

Documentation interactive : http://127.0.0.1:8000/docs
Health check : http://127.0.0.1:8000/health

Exemple de prédiction (curl) :
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_house.json