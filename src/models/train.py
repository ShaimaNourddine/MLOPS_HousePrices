import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("house_prices_experiment")

def train_model(X_train, y_train, X_test, y_test, params):
    """
    Entraîne un modèle RandomForest et log tout dans MLflow
    """
    with mlflow.start_run(run_name="rf_baseline"):
        # Log des hyperparamètres
        mlflow.log_params(params)
        
        # Entraînement
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        
        # Log des métriques
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Log d'un exemple d'entrée (obligatoire pour inférer la signature)
        input_example = X_test.iloc[[0]]  # une seule ligne comme exemple
        
        # Inférer et loguer la signature (très important pour l'API)
        signature = mlflow.models.infer_signature(X_test, y_pred)
        
        # Log du modèle avec signature et input example
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="house_prices_rf"
        )
        
        print(f"RMSE: {rmse:.2f} | R²: {r2:.4f} | MAE: {mae:.2f}")
        
        # Sauvegarde des colonnes attendues pour l'API (très important !)
        joblib.dump(list(X_train.columns), "deployment/api/expected_columns.joblib")
        print(f"Colonnes attendues sauvegardées : {len(X_train.columns)} colonnes")
        
        return model, rmse, r2, mae

if __name__ == "__main__":
    try:
        from src.data.load_data import load_and_split_data
        
        print("Chargement des données...")
        X_train, X_test, y_train, y_test = load_and_split_data(
            "data/raw/house_prices.csv"
        )
        
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1  # utilise tous les cœurs → plus rapide
        }
        
        print("Entraînement du modèle...")
        model, rmse, r2, mae = train_model(X_train, y_train, X_test, y_test, params)
        
        print("Entraînement terminé avec succès !")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution : {str(e)}")
        raise