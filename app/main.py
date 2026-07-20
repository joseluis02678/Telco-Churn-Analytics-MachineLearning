from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import CustomerData, PredictionResponse
from app.predict import ChurnPredictor
import os

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Microservicio para predicción de abandono de clientes",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo (ajusta la ruta según donde guardes tus modelos)
MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest.joblib")
predictor = ChurnPredictor(MODEL_PATH)

@app.get("/")
def read_root():
    return {
        "message": "Telco Churn API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        result = predictor.predict(customer.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
