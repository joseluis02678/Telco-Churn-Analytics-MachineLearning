from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class CustomerData(BaseModel):
    """
    Datos del cliente para predicción de churn.
    Debe coincidir EXACTAMENTE con las 13 variables usadas en training/train.py
    """
    
    # ==========================================
    # Variables numéricas originales (3)
    # El PCAFeatureEngineer las transformará internamente en:
    # - GastoVsTiempo
    # - tenure (se mantiene)
    # - ValorCliente
    # ==========================================
    tenure: float = Field(
        ..., 
        ge=0, 
        example=12.0, 
        description="Meses de permanencia del cliente"
    )
    MonthlyCharges: float = Field(
        ..., 
        ge=0, 
        example=70.35, 
        description="Cargo mensual en USD"
    )
    TotalCharges: float = Field(
        ..., 
        ge=0, 
        example=845.50, 
        description="Cargo total acumulado en USD"
    )
    
    # ==========================================
    # Variables categóricas (10)
    # Seleccionadas por Cramér's V >= 0.19
    # ==========================================
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., 
        example="Month-to-month",
        description="Tipo de contrato"
    )
    OnlineSecurity: Literal["Yes", "No"] = Field(
        ..., 
        example="No",
        description="Servicio de seguridad en línea"
    )
    TechSupport: Literal["Yes", "No"] = Field(
        ..., 
        example="No",
        description="Servicio de soporte técnico"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., 
        example="Fiber optic",
        description="Tipo de servicio de internet"
    )
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ] = Field(
        ..., 
        example="Electronic check",
        description="Método de pago"
    )
    OnlineBackup: Literal["Yes", "No"] = Field(
        ..., 
        example="No",
        description="Servicio de respaldo en línea"
    )
    DeviceProtection: Literal["Yes", "No"] = Field(
        ..., 
        example="No",
        description="Protección para dispositivos"
    )
    StreamingMovies: Literal["Yes", "No"] = Field(
        ..., 
        example="Yes",
        description="Acceso a películas en streaming"
    )
    StreamingTV: Literal["Yes", "No"] = Field(
        ..., 
        example="Yes",
        description="Acceso a TV en streaming"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., 
        example="Yes",
        description="Facturación electrónica"
    )

    # Configuración para Pydantic V2 (reemplaza a la antigua 'class Config')
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50,
                "Contract": "Month-to-month",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "StreamingMovies": "Yes",
                "StreamingTV": "Yes",
                "PaperlessBilling": "Yes"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Respuesta de la predicción de churn"""
    
    churn_prediction: str = Field(
        ..., 
        example="Yes", 
        description="Predicción binaria: Yes (churn) o No (permanece)"
    )
    churn_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.83, 
        description="Probabilidad de churn entre 0.0 y 1.0"
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., 
        example="High", 
        description="Nivel de riesgo del cliente"
    )
    model_version: str = Field(
        ..., 
        example="logreg_smotetomek_v1",
        description="Versión del modelo utilizado"
    )
    
    # Configuración para Pydantic V2
    model_config = ConfigDict(
        protected_namespaces=(),  # Elimina la advertencia de "model_"
        json_schema_extra={
            "example": {
                "churn_prediction": "Yes",
                "churn_probability": 0.83,
                "risk_level": "High",
                "model_version": "logreg_smotetomek_v1"
            }
        }
    )