# src/data/load_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_data(data_path: str, test_size: float = 0.2):
    """
    Charge et divise les données en train/test
    
    Args:
        data_path: Chemin vers le fichier CSV
        test_size: Proportion du test set
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Chargement des données depuis {data_path}")
    
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Nettoyage basique
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Remplir NaN
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Encoding catégorielles
    df = pd.get_dummies(df, drop_first=True)
    
    # Séparer features et target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data(
        "data/raw/house_prices.csv"
    )