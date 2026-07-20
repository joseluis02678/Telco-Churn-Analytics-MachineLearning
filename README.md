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

- [x] Machine Learning model trained & validated (Random Forest + SMOTE-Tomek)
- [x] Feature engineering pipeline replicated for inference
- [x] Model serialization (`.joblib` artifacts)
- [x] FastAPI REST API implementation
- [x] Pydantic data validation & schemas
- [x] Swagger UI auto-documentation
- [x] Docker containerization (`Dockerfile` & `docker-compose.yml`)
- [ ] CI/CD Pipeline (GitHub Actions)
- [ ] Cloud Deployment (Render / Railway)
- [ ] Monitoring & Logging (Prometheus / Grafana)

---

## 🏗️ System Architecture

The system is designed as a stateless microservice. The heavy lifting of training is done offline; the API only handles inference.

```text
┌─────────────────┐      POST /predict      ┌──────────────────────────────┐
│   Client /      │ ──────────────────────> │      FastAPI Server          │
│   Frontend      │ <────────────────────── │  (Uvicorn ASGI)              │
└─────────────────┘    JSON Response        └──────────────┬───────────────┘
                                                             │
                               ┌─────────────────────────────┼─────────────────────────────┐
                               ▼                              ▼                              ▼
                      ┌────────────────┐             ┌────────────────┐             ┌────────────────┐
                      │ Pydantic       │             │ Preprocessing  │             │ ML Model       │
                      │ Validation     │────────────>│ Pipeline       │────────────>│ (.joblib)      │
                      │ (schemas.py)   │             │ (Scaling, OHE) │             │ Random Forest  │
                      └────────────────┘             └────────────────┘             └────────────────┘
```

---

## 📁 Project Structure

The repository is organized following modern MLOps best practices, separating application logic, model artifacts, and infrastructure.

```text
Telco-Churn-Analytics-MachineLearning/
│
├── app/                        # Core application logic
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point & routes
│   ├── schemas.py              # Pydantic models for request/response
│   ├── predict.py              # Inference logic & model loading
│   └── utils.py                # Helper functions (e.g., preprocessing)
│
├── models/                     # Serialized ML artifacts
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   └── encoder.joblib
│
├── data/                       # Sample data for testing (optional)
│
├── tests/                      # Unit and integration tests
│   └── test_api.py
│
├── .github/
│   └── workflows/              # CI/CD pipelines (GitHub Actions)
│       └── deploy.yml
│
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore
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
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
```

### Example Response

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.83,
  "risk_level": "High"
}
```

---

## 🛠️ Tech Stack

| Category          | Technologies                                              |
| :----------------- | :---------------------------------------------------------- |
| Backend            | Python 3.10+, FastAPI, Uvicorn                              |
| Machine Learning    | Scikit-Learn, imbalanced-learn (SMOTE-Tomek), Pandas, NumPy |
| Validation         | Pydantic V2                                                  |
| Containerization    | Docker, Docker Compose                                       |
| CI/CD & Cloud       | GitHub Actions, Render / Railway (Pending)                  |

---

## ⚡ Quick Start (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Telco-Churn-Analytics-MachineLearning.git
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

### 4. Run the FastAPI server

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
- Add environment variables (e.g., `MODEL_PATH`).

**AWS ECS / GCP Cloud Run:**
- Push the Docker image to ECR / Artifact Registry.
- Deploy the container with auto-scaling rules.
