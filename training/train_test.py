"""
training/train.py — Pipeline de Churn Prediction (Telco)
==========================================================
Modelo ganador: Regresión Logística balanceada con SMOTE-Tomek

Outliers multivariantes: consenso de 4 métodos (>=2 coinciden)
    - Distancia de Mahalanobis (alpha=0.01)
    - Isolation Forest (contamination=0.02)
    - Test de Grubbs iterativo (alpha=0.01)
    - DBSCAN (eps = percentil 95 de k-NN distances)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2, t

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
)
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "TelcoCustomerChurn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_logreg_pipeline.joblib")

NUM_VARS_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_VARS = [  # seleccionadas por Cramér's V >= 0.19 frente a Churn
    "Contract", "OnlineSecurity", "TechSupport", "InternetService",
    "PaymentMethod", "OnlineBackup", "DeviceProtection",
    "StreamingMovies", "StreamingTV", "PaperlessBilling",
]
INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]


# ------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA
# ------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.set_index("customerID")
    for c in INTERNET_COLS:
        df[c] = df[c].replace("No internet service", "No")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def quick_eda(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nDistribución de Churn:")
    print(df["Churn"].value_counts(normalize=True).round(3))
    nulos = df.isna().sum()
    print("\nNulos por columna:\n", nulos[nulos > 0])


# ------------------------------------------------------------------
# 2. OUTLIERS MULTIVARIANTES — CONSENSO DE 4 MÉTODOS (solo en TRAIN)
# ------------------------------------------------------------------
def _mahalanobis_outliers(df_num: pd.DataFrame, alpha: float = 0.01) -> pd.Series:
    X = df_num.to_numpy()
    mean, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    d2 = np.sum((X - mean) @ inv_cov * (X - mean), axis=1)
    threshold = chi2.ppf(1 - alpha, df=X.shape[1])
    return pd.Series(d2 > threshold, index=df_num.index, name="Outlier_Mahalanobis")


def _isolation_forest_outliers(df_num: pd.DataFrame, contamination: float = 0.02) -> pd.Series:
    Xs = StandardScaler().fit_transform(df_num)
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                           random_state=RANDOM_STATE, n_jobs=-1)
    preds = iso.fit_predict(Xs)
    return pd.Series(preds == -1, index=df_num.index, name="Outlier_IF")


def _grubbs_outliers(df_num: pd.DataFrame, alpha: float = 0.01) -> pd.Series:
    def grubbs_iter(s: pd.Series) -> list:
        s = s.dropna().astype(float)
        idx = s.index.tolist()
        outs = []
        while len(idx) >= 3:
            vals = s.loc[idx]
            mean, std = vals.mean(), vals.std(ddof=1)
            if std == 0:
                break
            abs_dev = (vals - mean).abs()
            i_max = abs_dev.idxmax()
            G = abs_dev.max() / std
            n = len(vals)
            tcrit = t.ppf(1 - alpha / (2 * n), df=n - 2)
            Gcrit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit ** 2 / (n - 2 + tcrit ** 2))
            if G > Gcrit:
                outs.append(i_max)
                idx.remove(i_max)
            else:
                break
        return outs

    grubbs_idx = set()
    for col in df_num.columns:
        grubbs_idx.update(grubbs_iter(df_num[col]))
    return pd.Series(df_num.index.isin(grubbs_idx), index=df_num.index, name="Outlier_Grubbs")


def _dbscan_outliers(df_num: pd.DataFrame) -> pd.Series:
    Xs = StandardScaler().fit_transform(df_num)
    min_pts = max(4, 2 * Xs.shape[1] + 1)
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(Xs)
    distances, _ = nbrs.kneighbors(Xs)
    eps_opt = float(np.percentile(np.sort(distances[:, -1]), 95))
    labels = DBSCAN(eps=eps_opt, min_samples=min_pts).fit(Xs).labels_
    return pd.Series(labels == -1, index=df_num.index, name="Outlier_DBSCAN")


def keep_index_without_outliers(df_num: pd.DataFrame, min_votes: int = 2) -> pd.Index:
    """Consenso: se descarta un registro si >=min_votes métodos lo marcan outlier."""
    mahal = _mahalanobis_outliers(df_num)
    iso = _isolation_forest_outliers(df_num)
    grubbs = _grubbs_outliers(df_num)
    dbscan = _dbscan_outliers(df_num)

    combined = pd.concat([mahal, iso, grubbs, dbscan], axis=1).fillna(False)
    votes = combined.sum(axis=1)

    print(f"  Mahalanobis: {int(mahal.sum())} | Isolation Forest: {int(iso.sum())} "
          f"| Grubbs: {int(grubbs.sum())} | DBSCAN: {int(dbscan.sum())}")
    print(f"  Outliers por consenso (>= {min_votes} métodos): {int((votes >= min_votes).sum())}")

    return combined.index[votes < min_votes]


# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING: PCA -> ValorCliente / GastoVsTiempo
# ------------------------------------------------------------------
class PCAFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=RANDOM_STATE)

    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X[NUM_VARS_RAW]
        return pd.DataFrame(np.asarray(X), columns=NUM_VARS_RAW)

    def fit(self, X, y=None):
        Xdf = self._as_df(X)
        self.pca.fit(self.scaler.fit_transform(Xdf))
        return self

    def transform(self, X):
        Xdf = self._as_df(X)
        comps = self.pca.transform(self.scaler.transform(Xdf))
        return pd.DataFrame(
            {"GastoVsTiempo": comps[:, 1], "tenure": Xdf["tenure"].values, "ValorCliente": comps[:, 0]},
            index=Xdf.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(["GastoVsTiempo", "tenure", "ValorCliente"])


# ------------------------------------------------------------------
# 4. PIPELINE (Preprocesamiento + Balanceo + Modelo Ganador)
# ------------------------------------------------------------------
def build_pipeline() -> ImbPipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", PCAFeatureEngineer(), NUM_VARS_RAW),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_VARS),
    ])
    return ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("scale", StandardScaler()),
        ("balance", SMOTETomek(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


# ------------------------------------------------------------------
# 5. TABLA COMPLETA DE MÉTRICAS
# ------------------------------------------------------------------
def full_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)

    dfp = pd.DataFrame({"y": y_true, "score": y_proba}).sort_values("score", ascending=False)
    top10 = dfp.head(int(len(dfp) * 0.10))
    top20 = dfp.head(int(len(dfp) * 0.20))
    lift10 = top10["y"].mean() / dfp["y"].mean()
    lift20 = top20["y"].mean() / dfp["y"].mean()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": auc,
        "Gini": gini,
        "KS": ks,
        "Lift 10%": lift10,
        "Lift 20%": lift20,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ------------------------------------------------------------------
# 6. ENTRENAMIENTO Y EVALUACIÓN
# ------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\n=== Detección de outliers multivariantes (consenso, solo TRAIN) ===")
    keep_idx = keep_index_without_outliers(X_train[NUM_VARS_RAW], min_votes=2)
    X_train, y_train = X_train.loc[keep_idx], y_train.loc[keep_idx]
    print(f"  Train tras limpieza: {X_train.shape}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Matriz de Confusión (Test) ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Reporte de Clasificación (Test) ===")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    metrics_table = full_metrics(y_test, y_pred, y_proba)
    print("\n=== Tabla Completa de Métricas (Test) ===")
    for k, v in metrics_table.items():
        print(f"{k:20s}: {v:.4f}")

    return pipeline, (X_test, y_test)


# ------------------------------------------------------------------
# 7. SERIALIZACIÓN DEL MODELO
# ------------------------------------------------------------------
def save_model(pipeline: ImbPipeline, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\nModelo guardado en: {path}")


def load_model(path: str = MODEL_PATH) -> ImbPipeline:
    return joblib.load(path)


# ------------------------------------------------------------------
# 8. PREDICCIÓN SOBRE NUEVOS CLIENTES
# ------------------------------------------------------------------
def predict_churn(pipeline: ImbPipeline, new_data: dict) -> dict:
    df_new = pd.DataFrame([new_data])
    proba = float(pipeline.predict_proba(df_new)[0, 1])
    pred = "Yes" if proba >= 0.5 else "No"
    risk = "Alto" if proba >= 0.6 else "Medio" if proba >= 0.3 else "Bajo"
    return {"churn_prediction": pred, "churn_probability": round(proba, 4), "risk_level": risk}


# ------------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    quick_eda(df)

    model, (X_test, y_test) = train_and_evaluate(df)
    save_model(model)

    ejemplo = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 845.5,
    }
    print("\n=== Ejemplo de predicción ===")
    print(predict_churn(model, ejemplo))