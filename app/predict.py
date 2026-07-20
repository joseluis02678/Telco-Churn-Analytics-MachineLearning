import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()
        
    def preprocess(self, data: dict) -> np.ndarray:
        """Preprocesamiento básico - adaptar según tu pipeline real"""
        # Convertir a DataFrame
        df = pd.DataFrame([data])
        
        # Aquí debes replicar EXACTAMENTE el mismo preprocesamiento
        # que usaste en tu notebook de entrenamiento
        
        return df
    
    def predict(self, data: dict) -> dict:
        """Realizar predicción"""
        df = self.preprocess(data)
        
        # Predicción
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        # Determinar nivel de riesgo
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "model_version": "random_forest_v1"
        }
