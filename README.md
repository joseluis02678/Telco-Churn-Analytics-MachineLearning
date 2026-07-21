<h1 align="center">
  Telco Churn Prediction API
</h1>

<p align="center">
  <strong>Production-Ready Machine Learning Microservice</strong><br>
  REST API built with FastAPI, Docker, and Scikit-Learn for real-time customer churn prediction.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#-deployment-roadmap">Roadmap</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-rest-api-endpoints">API Endpoints</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-docker-deployment">Docker</a>
</p>

---

## 🚀 Deployment Roadmap

This branch is dedicated to transforming the trained Machine Learning model into a scalable, production-ready microservice.

- [x] Machine Learning model trained & validated (**Logistic Regression + SMOTE-Tomek**)
- [x] Multivariate outlier detection by consensus (Mahalanobis + Isolation Forest + Grubbs + DBSCAN)
- [x] Feature engineering pipeline replicated for inference (PCA + Cramér's V variable selection)
- [x] Model serialization (single `.joblib` pipeline artifact)
- [x] FastAPI REST API implementation
- [x] Pydantic data validation & schemas
- [x] Swagger UI auto-documentation
- [x] Docker containerization (`Dockerfile` & `docker-compose.yml`)
- [ ] CI/CD Pipeline (GitHub Actions)
- [ ] Cloud Deployment (Render / Railway)
- [ ] Monitoring & Logging (Prometheus / Grafana)

---

## 🏗️ System Architecture

The system is designed as a stateless microservice. Training happens offline (`training/train_test.py`); the API only handles inference by loading the single serialized pipeline.

```text
┌─────────────────┐      POST /predict      ┌──────────────────────────────┐
│   Client /      │ ──────────────────────> │      FastAPI Server          │
│   Frontend      │ <────────────────────── │  (Uvicorn ASGI)              │
└─────────────────┘    JSON Response        └──────────────┬───────────────┘
                                                             │
                                                             ▼
                                                ┌──────────────────────────┐
                                                │ ChurnPredictor           │
                                                │ (app/predict.py)         │
                                                │                          │
                                                │ Loads a single sklearn/  │
                                                │ imblearn Pipeline:       │
                                                │  1. PCA feature engineer │
                                                │  2. OneHotEncoder        │
                                                │  3. StandardScaler       │
                                                │  4. SMOTETomek (train    │
                                                │     only, skipped at    │
                                                │     inference)           │
                                                │  5. LogisticRegression   │
                                                │                          │
                                                │ churn_logreg_pipeline    │
                                                │       .joblib            │
                                                └──────────────────────────┘
```

**Winning model:** Logistic Regression (`class_weight="balanced"`) trained on SMOTE-Tomek-balanced data. Chosen over Random Forest for its higher recall on the churn class (~0.79 vs ~0.49), which better serves the business goal of catching real churners even at the cost of some false positives.

| Metric (Test) | Value |
| :------------- | :---- |
| ROC AUC        | ~0.839 |
| Recall (Churn) | ~0.79 |
| Precision (Churn) | ~0.50 |
| Gini           | ~0.68 |
| KS             | ~0.52 |

---

## 📁 Project Structure

```text
Telco-Churn-Analytics-MachineLearning/
│
├── app/                          # Core application logic (PRODUCTION)
│   ├── __init__.py
│   ├── main.py                   # FastAPI entry point & routes
│   ├── predict.py                # ChurnPredictor: loads .joblib & runs inference
│   └── schemas.py                # Pydantic models for request/response
│
├── training/                     # Offline training pipeline
│   ├── train_test.py             # Full pipeline: EDA -> outliers -> PCA ->
│   │                              # SMOTE-Tomek -> Logistic Regression -> save
│   └── inspect_model.py          # Utility to load & inspect the saved pipeline
│
├── models/                       # Serialized ML artifact
│   └── churn_logreg_pipeline.joblib   # Single pipeline: preprocessing + model
│
├── data/                         # Raw dataset
│   └── TelcoCustomerChurn.csv
│
├── notebooks/                    # Exploratory analysis / academic backup
│   ├── Trabajo final - Ciencia de datos_revision.pdf
│   ├── Trabajo_final_Módulo_5_3_10.ipynb
│   └── Trabajo_final_Módulo_5_4_2.ipynb
│
├── Dockerfile                    # Container build instructions
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🌐 REST API Endpoints

Once the server is running, the API exposes the following endpoints:

| Method | Endpoint   | Description                                                        | Auth |
| :----- | :--------- | :------------------------------------------------------------------ | :--- |
| GET    | `/`        | Root endpoint, API status & version                                 | No   |
| GET    | `/health`  | Liveness/Readiness check for orchestrators                          | No   |
| POST   | `/predict` | Main inference endpoint. Expects customer features, returns churn probability & risk level. | No   |
| GET    | `/docs`    | Interactive Swagger UI documentation                                | No   |
| GET    | `/redoc`   | Alternative ReDoc documentation                                     | No   |

### Example Request (`/predict`)

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
```

### Example Response

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.8726,
  "risk_level": "Alto"
}
```

---

## 🛠️ Tech Stack

| Category          | Technologies                                              |
| :----------------- | :---------------------------------------------------------- |
| Backend            | Python 3.10+, FastAPI, Uvicorn                              |
| Machine Learning    | Scikit-Learn, imbalanced-learn (SMOTE-Tomek), Pandas, NumPy, SciPy |
| Validation         | Pydantic V2                                                  |
| Containerization    | Docker, Docker Compose                                       |
| CI/CD & Cloud       | GitHub Actions, Render / Railway (Pending)                  |

---

## ⚡ Quick Start (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/joseluis02678/Telco-Churn-Analytics-MachineLearning.git
cd Telco-Churn-Analytics-MachineLearning
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (generates the `.joblib` artifact)

```bash
python training/train_test.py
```

This reads `data/TelcoCustomerChurn.csv`, runs the full pipeline (EDA, outlier consensus, PCA, SMOTE-Tomek, Logistic Regression), and saves `models/churn_logreg_pipeline.joblib`.

### 5. Run the FastAPI server

```bash
uvicorn app.main:app --reload
```

> 💡 **Tip:** Open your browser and go to `http://localhost:8000/docs` to interact with the auto-generated Swagger UI.

---

## 🐳 Docker Deployment

The application is fully containerized for consistent deployment across any environment.

### Build the image

```bash
docker build -t telco-churn-api .
```

### Run the container

```bash
docker run -p 8000:8000 telco-churn-api
```

### Or use Docker Compose (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

---

## ☁️ Cloud Deployment Strategy

This project is designed to be easily deployed to PaaS providers:

**Render / Railway:**
- Connect the GitHub repository.
- Set the build command: `pip install -r requirements.txt`
- Set the start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add environment variables (e.g., `MODEL_PATH=models/churn_logreg_pipeline.joblib`).

**AWS ECS / GCP Cloud Run:**
- Push the Docker image to ECR / Artifact Registry.
- Deploy the container with auto-scaling rules.
