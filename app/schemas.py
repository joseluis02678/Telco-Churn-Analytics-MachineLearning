from pydantic import BaseModel, Field
from typing import Optional

class CustomerData(BaseModel):
    # Variables numéricas
    tenure: float = Field(..., example=12.0, description="Meses de permanencia")
    MonthlyCharges: float = Field(..., example=70.35, description="Cargo mensual")
    TotalCharges: float = Field(..., example=845.50, description="Cargo total acumulado")
    
    # Variables categóricas
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="0 = No, 1 = Sí")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    PhoneService: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")

class PredictionResponse(BaseModel):
    churn_prediction: str = Field(..., example="Yes", description="Yes o No")
    churn_probability: float = Field(..., example=0.83, description="Probabilidad entre 0.0 y 1.0")
    risk_level: str = Field(..., example="High", description="Low, Medium, o High")
    model_version: str = Field(..., example="random_forest_v1")
