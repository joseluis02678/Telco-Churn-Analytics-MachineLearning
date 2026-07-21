import os
import sys
import joblib

# Aseguramos que Python pueda encontrar train.py en la carpeta training
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necesario para reconstruir el objeto PCAFeatureEngineer
from training.train import PCAFeatureEngineer  # noqa: F401

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_logreg_pipeline.joblib")

# Verificar que el modelo existe
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: No se encontró el modelo en: {MODEL_PATH}")
    print("   Asegúrate de haber ejecutado: python training/train.py")
    sys.exit(1)

# Cargar el pipeline
pipeline = joblib.load(MODEL_PATH)

print("✅ Modelo cargado exitosamente\n")
print("Pasos del pipeline:")
for step_name, step_obj in pipeline.named_steps.items():
    print(f"  - {step_name}: {type(step_obj).__name__}")

print("\n Coeficientes del modelo (Logistic Regression):")
clf = pipeline.named_steps["clf"]
print(f"  Intercepto: {clf.intercept_}")
print(f"  Coeficientes (shape): {clf.coef_.shape}")
print(f"\n  Primeros 10 coeficientes:")
for i, coef in enumerate(clf.coef_[0][:10]):
    print(f"    Feature {i}: {coef:.4f}")