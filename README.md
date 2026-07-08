# 📉 Telco Customer Churn Prediction & Segmentation

## 🎯 Contexto y Problema de Negocio
Durante el último mes, una empresa del sector de telecomunicaciones experimentó un incremento en su ratio de *churn* (abandono de clientes) de +0.5 puntos porcentuales, pasando de un 2% a un 2.5% debido al impacto del COVID-19. Dado que el costo de adquisición de un nuevo cliente es significativamente mayor que el de retención, este proyecto busca identificar patrones predictivos para optimizar estrategias de fidelización.

## 🛠️ Tecnologías y Herramientas
* **Lenguaje:** Python 
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, SciPy
* **Visualization:** Matplotlib, Seaborn

## 🧠 Metodología y Flujo del Proyecto

### 1. Análisis Exploratorio (EDA) y Preprocesamiento
* Limpieza de datos e imputación de valores faltantes (`TotalCharges`).
* **Detección de Outliers:** Identificación y eliminación de valores anómalos mediante el consenso de múltiples métodos (Distancia de Mahalanobis, Isolation Forest, Test de Grubbs, DBSCAN).

### 2. Feature Engineering y Selección
* **Asociación:** Evaluación mediante V de Cramér.
* **Reducción de Dimensionalidad:** Implementación de **PCA** para variables numéricas (creando componentes: "ValorCliente" y "GastoVsTiempo") y **MCA** para variables categóricas.

### 3. Segmentación (Clustering)
* Aplicación de **K-Means** ($k=4$) sobre el espacio reducido de características para identificar perfiles de riesgo. 
* El **Cluster 2** resultó ser el de mayor riesgo, con una tasa de *churn* cercana al 46%.

### 4. Modelado Predictivo
* Entrenamiento de un **Random Forest Balanceado** enfocado en maximizar la capacidad de detección del segmento de alto riesgo.

## 📊 Resultados Clave
* **ROC AUC:** 0.84 (fuerte capacidad de discriminación).
* **Lift (Top 10%):** 2.89x (el modelo es casi 3 veces más efectivo que el azar en el primer decil).
* **Variables Predictoras:** La relación *GastoVsTiempo* y el tipo de contrato son los principales determinantes de la fuga.

## 🚀 Cómo ejecutar este proyecto
1. Clona este repositorio: `git clone https://github.com/tu-usuario/Telco-Churn-Analytics-MachineLearning.git`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Explora el notebook en la carpeta `notebooks/`.

## 👥 Autores
* **Jose Luis Garay Ramos**
* **Diana Chavez**
