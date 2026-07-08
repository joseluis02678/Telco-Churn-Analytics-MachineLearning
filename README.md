<p align="center">
Customer Segmentation • Predictive Analytics • Business Intelligence
</p>

<p align="center">
Machine Learning • Customer Analytics • Customer Segmentation • Business Intelligence
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas">
<img src="https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn">
<img src="https://img.shields.io/badge/imbalanced--learn-SMOTE--Tomek-red">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-blue">
<img src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">
  <a href="#-descripción">Descripción</a> •
  <a href="#-contexto-del-problema">Contexto</a> •
  <a href="#-objetivos">Objetivos</a> •
  <a href="#-workflow-del-proyecto">Workflow</a> •
  <a href="#-resultados-clave">Resultados</a> •
  <a href="#-cómo-ejecutar-el-proyecto">Ejecución</a>
</p>

---

## 📖 Descripción

Este proyecto desarrolla una solución integral de **Machine Learning** para analizar y predecir el abandono de clientes (*Customer Churn*) en una empresa de telecomunicaciones.

Además del modelo predictivo, se implementa una estrategia de **segmentación de clientes** mediante técnicas de reducción de dimensionalidad (PCA + MCA) y clustering (K-Means), con el objetivo de identificar perfiles de clientes y apoyar la toma de decisiones orientadas a la retención.

El proyecto cubre todas las etapas de un flujo de Ciencia de Datos, desde el entendimiento del negocio hasta la construcción, evaluación e interpretación comparativa de tres modelos de clasificación.

---

## 📌 Contexto del problema

Durante el periodo analizado, la empresa experimentó un incremento en la tasa de abandono de clientes debido, en parte, al impacto generado por la pandemia del COVID-19.

Dado que adquirir un nuevo cliente resulta significativamente más costoso que conservar uno existente, surge la necesidad de un modelo capaz de **identificar anticipadamente** a los clientes con mayor probabilidad de abandonar la compañía, así como de segmentarlos para diseñar estrategias de retención más eficientes y focalizadas.

---

## 🎯 Objetivos

- Analizar los factores asociados al abandono de clientes (*churn*).
- Construir y comparar modelos predictivos de clasificación binaria.
- Identificar segmentos de clientes con perfiles de riesgo diferenciados.
- Generar información accionable para el área de marketing y retención.
- Formular recomendaciones de negocio basadas en evidencia estadística.

---

## 🗂️ Dataset

- **Fuente:** Telco Customer Churn (estructura tipo IBM/Kaggle).
- **Variable objetivo:** `Churn` (Yes / No).
- **Variables:** información demográfica (género, adultos mayores, dependientes), contractuales (tipo de contrato, método de pago, facturación electrónica) y de consumo (`tenure`, `MonthlyCharges`, `TotalCharges`, servicios contratados).
- **Partición:** división Train/Test aplicada antes de cualquier tratamiento, con reproducción idéntica del pipeline de limpieza en ambos conjuntos para evitar fuga de información (*data leakage*).

---

## 🛠 Tecnologías utilizadas

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="50"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="50"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="50"/>
<img src="https://matplotlib.org/_static/images/logo2.svg" width="110"/>
</p>

| Categoría | Librerías |
|---|---|
| Manipulación de datos | `pandas`, `numpy` |
| Modelado / preprocesamiento | `scikit-learn` |
| Balanceo de clases | `imbalanced-learn` (SMOTE, SMOTE-Tomek) |
| Análisis multivariado categórico | `prince` (MCA) |
| Estadística | `scipy` |
| Scorecards / WOE-IV | `scorecardpy` |
| Visualización | `matplotlib`, `seaborn` |

---

## 🔄 Workflow del proyecto

```text
Business Understanding + Machine Learning Canvas
                │
                ▼
        Train / Test Split
                │
                ▼
   Exploratory Data Analysis (EDA)
                │
                ▼
  Data Cleaning & Missing Value Imputation
                │
                ▼
      Multivariate Outlier Detection
 (Mahalanobis • Isolation Forest • Grubbs • DBSCAN)
                │
                ▼
   Association Analysis (Cramér's V) + IV / WOE
                │
                ▼
     Dimensionality Reduction (PCA + MCA)
                │
                ▼
   Customer Segmentation (K-Means, k=4)
                │
                ▼
 Modeling: Logistic Regression • CART • Random Forest
        (baseline vs. SMOTE-Tomek balanced)
                │
                ▼
Model Evaluation (ROC AUC • Gini • KS • Lift • Confusion Matrix)
                │
                ▼
      Business Insights & Recommendations
```

---

## 📂 Contenido del notebook

### 1. Comprensión del negocio
Definición del problema, diccionario de variables y desarrollo del **Machine Learning Canvas** que guía el proyecto.

### 2. Análisis exploratorio de datos (EDA)
Estadísticas descriptivas y visualizaciones univariadas para variables numéricas y categóricas, además del análisis del balance de la variable objetivo.

### 3. Limpieza y preparación de datos
Corrección de tipos, imputación de valores faltantes (con análisis de dependencia previo, evitando reglas automáticas ingenuas), codificación de variables categóricas y escalamiento.

### 4. Detección de valores atípicos multivariados
Se aplican y comparan cuatro métodos, tomando la decisión final por **consenso** entre ellos:
- Distancia de Mahalanobis
- Isolation Forest
- Test de Grubbs
- DBSCAN

### 5. Ingeniería de características y asociación
Creación de variables derivadas (p. ej. relación gasto/tiempo, valor del cliente) y evaluación de asociación con `Churn` mediante el **coeficiente V de Cramér** e **Information Value (IV)**.

### 6. Reducción de dimensionalidad
- **PCA** sobre variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`), generando componentes interpretables como *valor del cliente* y *relación gasto vs. permanencia*.
- **MCA** sobre variables categóricas (tipo de contrato, servicios, forma de pago, etc.).

### 7. Segmentación de clientes
**K-Means** sobre las componentes de PCA + MCA. El número óptimo de clusters (**k = 4**) se valida con el método del codo y el coeficiente de silueta, y se confirma su estabilidad al proyectar el modelo entrenado sobre el conjunto de test.

### 8. Modelado predictivo
Se entrenan y comparan tres algoritmos, tanto en su versión original (datos desbalanceados) como balanceada con **SMOTE-Tomek**:
- Regresión Logística Binaria
- Árbol de Decisión (CART), con poda vía `ccp_alpha`
- Random Forest

### 9. Evaluación del modelo
Comparación sistemática mediante ROC AUC, Gini, KS, Lift, matriz de confusión y métricas por clase, evaluando en particular el trade-off entre precisión y recall sobre la clase minoritaria (*churn*).

---

## 📊 Resultados clave

**Segmentación de clientes (K-Means, k = 4):**

| Cluster | % Churn | Perfil |
|---|---:|---|
| 0 | ~1% | Clientes muy fidelizados, contratos largos |
| 1 | ~13–15% | Riesgo bajo a moderado |
| 2 | **~45–46%** | **Alto riesgo — segmento prioritario de retención** |
| 3 | ~11–14% | Estables con señales de riesgo |

La estructura de los cuatro clusters se mantuvo estable al validarse sobre el conjunto de test, confirmando la robustez de la segmentación.

**Modelo predictivo — mejor desempeño (Random Forest balanceado):**

| Métrica | Train | Test |
|---|---:|---:|
| ROC AUC | ~0.88 | **~0.84** |
| Accuracy | ~0.80 | ~0.79 |

> El modelo mantiene un desempeño estable entre train y test (sin sobreajuste evidente) y conserva una buena capacidad para identificar clientes en riesgo de churn tras el balanceo con SMOTE-Tomek.

Se comparó además el desempeño de Regresión Logística y CART (con y sin balanceo), observando en todos los casos la mejora en el *recall* de la clase minoritaria al aplicar SMOTE-Tomek, a costa de una ligera reducción en *precision* — un trade-off relevante para el diseño de campañas de retención.

---

## ⭐ Técnicas implementadas

- Machine Learning Canvas
- Exploratory Data Analysis (EDA)
- Missing Value Imputation (basada en análisis de dependencia)
- Detección multivariada de outliers: Mahalanobis, Isolation Forest, Grubbs, DBSCAN
- Feature Engineering
- Cramér's V / Information Value (IV) / Weight of Evidence (WOE)
- Principal Component Analysis (PCA)
- Multiple Correspondence Analysis (MCA)
- K-Means Clustering (validado con Elbow + Silhouette)
- Regresión Logística, CART (con poda por `ccp_alpha`), Random Forest
- Balanceo de clases con SMOTE-Tomek
- Evaluación: ROC AUC, Gini, KS, Lift, Matriz de Confusión, MCC

---

## 💼 Valor para el negocio

- Identifica clientes con alta probabilidad de abandono antes de que ocurra.
- Prioriza campañas de retención sobre el segmento de mayor riesgo (~46% de churn).
- Optimiza la asignación de recursos comerciales mediante segmentación accionable.
- Provee una base analítica reproducible para futuras iteraciones del modelo.

---

## 📁 Estructura del repositorio

```text
Telco-Churn-Analytics/
│
├── data/
├── notebooks/
│   └── Trabajo_final_Módulo_5_4_2.ipynb
├── images/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Cómo ejecutar el proyecto

```bash
git clone https://github.com/tu-usuario/Telco-Churn-Analytics.git
cd Telco-Churn-Analytics
pip install -r requirements.txt
```

Luego abre el notebook ubicado en `notebooks/` y ejecuta las celdas en orden.

---

## 👨‍💻 Autores

**José Luis Garay Ramos**
Estudiante de Ingeniería Estadística e Informática
Universidad Nacional Agraria La Molina

**Diana Valeri Chavez Palomino**
Estudiante de Ingeniería de Software
Universidad Nacional de Ingeniería

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
