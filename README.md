<h1 align="center">
📉 Telco Customer Churn Prediction & Customer Segmentation
</h1>

<p align="center">
Machine Learning • Customer Analytics • Customer Segmentation • Business Intelligence
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas">
<img src="https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-blue">
<img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

### 📖 Descripción

Este proyecto desarrolla una solución integral de **Machine Learning** para analizar y predecir el abandono de clientes (*Customer Churn*) en una empresa de telecomunicaciones.

Además del modelo predictivo, se implementa una estrategia de segmentación mediante técnicas de reducción de dimensionalidad y clustering con el objetivo de identificar perfiles de clientes y apoyar la toma de decisiones orientadas a la retención.

El proyecto cubre todas las etapas de un flujo de Ciencia de Datos, desde el entendimiento del problema de negocio hasta la construcción, evaluación e interpretación del modelo final.

---

### 📌 Contexto del problema

Durante el periodo analizado, la empresa experimentó un incremento en la tasa de abandono de clientes, pasando del **2% al 2.5%** debido al impacto generado por la pandemia del COVID-19.

Considerando que adquirir un nuevo cliente resulta significativamente más costoso que conservar uno existente, surge la necesidad de desarrollar un modelo capaz de identificar anticipadamente a los clientes con mayor probabilidad de abandonar la compañía y generar estrategias de retención más eficientes.

---

### 🎯 Objetivos

- Analizar los factores relacionados con el abandono de clientes.
- Construir un modelo predictivo de Customer Churn.
- Identificar segmentos de clientes con características similares.
- Obtener información útil para apoyar decisiones de negocio.
- Generar recomendaciones para estrategias de fidelización.

---

### 🛠 Tecnologías utilizadas

<p align="center">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="55"/>

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="55"/>

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="55"/>

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="55"/>

<img src="https://matplotlib.org/_static/images/logo2.svg" width="120"/>

</p>

- Python
- Pandas
- NumPy
- Scikit-Learn
- SciPy
- Matplotlib
- Seaborn

---

## 🔄 Workflow del Proyecto

```text
Business Understanding
        │
        ▼
Machine Learning Canvas
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Missing Value Imputation
        │
        ▼
Outlier Detection
(Mahalanobis • Isolation Forest • Grubbs • DBSCAN)
        │
        ▼
Feature Engineering
        │
        ▼
Association Analysis
(Cramér's V)
        │
        ▼
Dimensionality Reduction
(PCA + MCA)
        │
        ▼
Customer Segmentation
(K-Means)
        │
        ▼
Balanced Random Forest
        │
        ▼
Model Evaluation
(ROC AUC • Lift • Confusion Matrix)
        │
        ▼
Business Insights & Conclusions
```

---

## 📂 Contenido del Notebook

### 📌 1. Comprensión del negocio

Se define el problema de negocio, los objetivos del proyecto y el impacto económico que representa el abandono de clientes.

También se desarrolla el **Machine Learning Canvas**, estableciendo los elementos que guían todo el proceso analítico.

---

### 📌 2. Análisis Exploratorio de Datos (EDA)

Durante esta etapa se estudia el comportamiento general del conjunto de datos mediante estadísticas descriptivas y visualizaciones.

Se analizan:

- Distribución de variables.
- Variables numéricas y categóricas.
- Relación entre variables.
- Balance de la variable objetivo.
- Patrones asociados al abandono.

---

### 📌 3. Limpieza y preparación de datos

Antes del modelado se realizan diferentes tareas de preprocesamiento:

- Corrección de tipos de datos.
- Tratamiento de valores faltantes.
- Imputación de **TotalCharges**.
- Codificación de variables categóricas.
- Escalamiento de variables numéricas.

---

### 📌 4. Detección de valores atípicos

Se emplean diferentes algoritmos para detectar observaciones anómalas.

Los métodos utilizados fueron:

- Distancia de Mahalanobis
- Isolation Forest
- Test de Grubbs
- DBSCAN

La decisión final se obtiene considerando el consenso entre los diferentes métodos.

---

### 📌 5. Ingeniería de características

Se generan nuevas variables que representan de mejor manera el comportamiento de los clientes y aumentan la capacidad predictiva del modelo.

Además, se evalúa la asociación entre variables mediante el coeficiente **V de Cramér**.

---

### 📌 6. Reducción de dimensionalidad

Para sintetizar la información se implementan dos técnicas complementarias.

#### PCA

Aplicado sobre variables numéricas para construir componentes relacionados con:

- Valor del cliente.
- Relación entre gasto y permanencia.

#### MCA

Aplicado sobre variables categóricas para representar las categorías en un espacio reducido de dimensiones.

---

### 📌 7. Segmentación de clientes

Se implementa un modelo **K-Means** sobre las componentes obtenidas mediante PCA y MCA.

El análisis permite identificar **cuatro segmentos** de clientes con perfiles claramente diferenciados.

Uno de los grupos presenta una tasa de abandono cercana al **46%**, convirtiéndose en el segmento prioritario para estrategias de retención.

---

### 📌 8. Modelo predictivo

Se desarrolla un modelo de **Balanced Random Forest**, seleccionado por su capacidad para trabajar con clases desbalanceadas.

El proceso incluye:

- División Train/Test.
- Entrenamiento.
- Predicción.
- Evaluación del desempeño.

---

### 📌 9. Evaluación del modelo

El desempeño del modelo se evalúa utilizando diferentes métricas de clasificación.

| Métrica | Resultado |
|---------|----------:|
| ROC AUC | **0.84** |
| Lift (Top 10%) | **2.89** |
| Clusters | **4** |
| Cluster de mayor riesgo | **46% Churn** |

---

### ⭐ Técnicas implementadas

- Machine Learning Canvas
- Exploratory Data Analysis (EDA)
- Missing Value Imputation
- Outlier Detection
- Mahalanobis Distance
- Isolation Forest
- Grubbs Test
- DBSCAN
- Feature Engineering
- Cramér's V
- Principal Component Analysis (PCA)
- Multiple Correspondence Analysis (MCA)
- K-Means Clustering
- Balanced Random Forest
- Model Evaluation
- Lift Analysis
- Business Insights

---

### 💼 Valor para el negocio

Este proyecto permite:

- Detectar clientes con alta probabilidad de abandono.
- Priorizar campañas de retención.
- Optimizar recursos comerciales.
- Identificar segmentos de clientes con diferentes perfiles de riesgo.
- Apoyar la toma de decisiones basada en datos.

---

### 📁 Estructura del repositorio

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

### 🚀 Cómo ejecutar el proyecto

```bash
git clone https://github.com/tu-usuario/Telco-Churn-Analytics.git
```

```bash
pip install -r requirements.txt
```

Finalmente, abre el notebook ubicado en la carpeta **notebooks** y ejecuta las celdas en orden.

---

### 👨‍💻 Autores

**Jose Luis Garay Ramos**

Estudiante de Ingeniería Estadística e Informática  
Universidad Nacional Agraria La Molina

**Diana Chavez**

Estudiante de Ingenería de Software
Universidad Nacional de Ingeniería
