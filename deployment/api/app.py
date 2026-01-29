# deployment/api/app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
from typing import Dict, Any

app = FastAPI(
    title="House Prices Prediction API",
    description="API de prédiction de prix immobiliers (RandomForest via MLflow)",
    version="1.0.0"
)

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "house_prices_rf"
MODEL_STAGE = "None"  # change en "Production" plus tard quand tu promouvois

# Chargement du modèle (fait une seule fois au démarrage)
def load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modèle chargé depuis : {model_uri}")
        return model
    except Exception as e:
        raise RuntimeError(f"Erreur chargement modèle : {str(e)}")

model = load_model()

# Chargement des colonnes attendues (créé par train.py)
try:
    EXPECTED_COLUMNS = joblib.load(r"C:\Users\SBenNourddin\mlops-project\deployment\api\expected_columns.joblib")
    print(f"Colonnes attendues : {len(EXPECTED_COLUMNS)} features")
except Exception as e:
    raise RuntimeError(f"Erreur chargement expected_columns.joblib : {str(e)}")

class HouseFeatures(BaseModel):
    MSSubClass: int = 60
    LotArea: int = 8450
    OverallQual: int = 7
    OverallCond: int = 5
    YearBuilt: int = 2003
    YearRemodAdd: int = 2003
    TotalBsmtSF: float = 856.0
    GrLivArea: float = 1710.0
    FullBath: int = 2
    HalfBath: int = 1
    BedroomAbvGr: int = 3
    TotRmsAbvGrd: int = 8
    Fireplaces: int = 0
    GarageCars: float = 2.0
    GarageArea: float = 548.0
    # Tu peux ajouter d'autres features si tu veux (mais l'API complète le reste à 0)

def preprocess_input(input_dict: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])
    
    # Remplissage NaN / valeurs par défaut
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("None")
    
    # One-hot encoding (comme dans load_data.py)
    df = pd.get_dummies(df, drop_first=True)
    
    # Réaligner exactement sur les colonnes d'entraînement
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    return df

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "features_count": len(EXPECTED_COLUMNS)}

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        input_df = preprocess_input(features.dict())
        prediction = model.predict(input_df)[0]
        
        return {
            "predicted_price": round(float(prediction), 2),
            "currency": "USD",
            "model_version": MODEL_NAME,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)