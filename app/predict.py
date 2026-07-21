import sys
import os
import __main__
import joblib
import pandas as pd
from typing import Dict, Any

# 1. Asegurar que la carpeta raíz del proyecto esté en el path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 2. Importar la clase personalizada
try:
    from training.train import PCAFeatureEngineer
except ImportError:
    try:
        from training.train_test import PCAFeatureEngineer
    except ImportError:
        raise ImportError("No se pudo encontrar PCAFeatureEngineer. Asegúrate de que el archivo exista en la carpeta 'training/'.")

# 3. 🚨 TRUCO CRÍTICO PARA JOBLIB 🚨
# Como la clase se guardó cuando el script se ejecutó como '__main__',
# joblib buscará la clase en el módulo '__main__' actual.
# Debemos registrarla allí manualmente para que joblib la encuentre.
setattr(__main__, 'PCAFeatureEngineer', PCAFeatureEngineer)


class ChurnPredictor:
    def __init__(self, model_path: str):
        print(f"🔄 Cargando modelo desde: {model_path}")
        # Ahora joblib podrá reconstruir el objeto sin errores
        self.pipeline = joblib.load(model_path)
        print("✅ Modelo cargado exitosamente.")
        
        # Las 13 columnas exactas que espera el modelo entrenado
        self.expected_columns = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'OnlineSecurity', 'TechSupport', 'InternetService',
            'PaymentMethod', 'OnlineBackup', 'DeviceProtection',
            'StreamingMovies', 'StreamingTV', 'PaperlessBilling'
        ]
        
    def preprocess(self, data: Dict[str, Any]) -> pd.DataFrame:
        # Asegurar tipos correctos para las variables numéricas
        data['tenure'] = float(data['tenure'])
        data['MonthlyCharges'] = float(data['MonthlyCharges'])
        data['TotalCharges'] = float(data['TotalCharges'])
        
        # Crear DataFrame con una sola fila
        df = pd.DataFrame([data])
        
        # Verificar que tenga todas las columnas esperadas
        for col in self.expected_columns:
            if col not in df.columns:
                raise ValueError(f"Falta la columna requerida: {col}")
        
        # Seleccionar solo las columnas necesarias en el orden correcto
        return df[self.expected_columns]
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = self.preprocess(data)
        
        # El pipeline se encarga de TODO: PCA, OneHot, Scaler y Predicción
        prediction = self.pipeline.predict(df)[0]
        probability = float(self.pipeline.predict_proba(df)[0][1])
        
        # Determinar nivel de riesgo
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "model_version": "logreg_smotetomek_v1"
        }