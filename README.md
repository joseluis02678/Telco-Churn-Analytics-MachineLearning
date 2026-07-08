## 📉 Telco Customer Churn Prediction & Customer Segmentation

Este proyecto tiene como objetivo analizar el comportamiento de los clientes de una empresa de telecomunicaciones para comprender las principales causas del abandono (*Customer Churn*) y desarrollar un modelo capaz de identificar a los clientes con mayor riesgo de cancelar el servicio.

Además de construir un modelo predictivo, se realizó un proceso de segmentación de clientes que permite identificar distintos perfiles de comportamiento y facilitar la toma de decisiones para campañas de retención.

---

## 📌 Contexto del problema

Durante el periodo analizado, la empresa experimentó un incremento en su tasa de abandono de clientes, pasando del **2% al 2.5%** como consecuencia del impacto generado por la pandemia del COVID-19.

Dado que captar un nuevo cliente representa un costo considerablemente mayor que conservar uno existente, el área de negocio requiere una solución basada en datos que permita anticipar el abandono y priorizar acciones de fidelización sobre los clientes con mayor riesgo.

---

## 🎯 Objetivos

Los principales objetivos del proyecto fueron:

* Analizar los factores asociados al abandono de clientes.
* Construir un modelo de Machine Learning para predecir el churn.
* Identificar segmentos de clientes con comportamientos similares.
* Obtener información que facilite el diseño de estrategias comerciales y de retención.

---

## 🛠 Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* SciPy
* Matplotlib
* Seaborn

---

## 📂 Contenido del proyecto

El notebook desarrolla un flujo completo de Ciencia de Datos, desde la comprensión del problema hasta la evaluación del modelo final.

### Comprensión del negocio

Se define el problema, los objetivos del proyecto y el impacto que tiene el abandono de clientes dentro de la organización. Además, se desarrolla el Machine Learning Canvas como guía para todo el proceso analítico.

### Análisis exploratorio de datos (EDA)

Se estudia la distribución de las variables, la calidad de los datos y las relaciones existentes entre ellas mediante estadísticas descriptivas y visualizaciones.

Durante esta etapa también se identifican patrones relevantes relacionados con el abandono de clientes.

### Limpieza y preparación de datos

Antes de construir los modelos se realizaron diferentes tareas de preprocesamiento, entre ellas:

* Corrección de tipos de datos.
* Tratamiento de valores faltantes.
* Imputación de la variable **TotalCharges**.
* Codificación de variables categóricas.
* Escalamiento de variables numéricas.

### Detección de valores atípicos

Para reducir el efecto de observaciones anómalas se empleó una estrategia basada en varios métodos de detección de outliers:

* Distancia de Mahalanobis
* Isolation Forest
* Test de Grubbs
* DBSCAN

La decisión final se tomó considerando el consenso entre estos métodos.

### Ingeniería de características

Se construyeron nuevas variables con el objetivo de mejorar la capacidad predictiva del modelo y representar mejor el comportamiento de los clientes.

También se evaluó la relación entre variables utilizando el coeficiente **V de Cramér**.

### Reducción de dimensionalidad

Con el fin de resumir la información del conjunto de datos se aplicaron dos técnicas complementarias.

**PCA (Principal Component Analysis)** para las variables numéricas, obteniendo componentes relacionados con el valor del cliente y la relación entre gasto y permanencia.

**MCA (Multiple Correspondence Analysis)** para representar las variables categóricas en un espacio de menor dimensión.

### Segmentación de clientes

Utilizando las componentes obtenidas mediante PCA y MCA se implementó un modelo de **K-Means** para identificar grupos de clientes con características similares.

Como resultado se obtuvieron cuatro segmentos claramente diferenciados, destacando un grupo cuya tasa de abandono alcanzó aproximadamente el **46%**, convirtiéndose en el principal objetivo para estrategias de retención.

### Modelo predictivo

Finalmente se entrenó un modelo de **Balanced Random Forest**, seleccionado por su buen desempeño frente al desbalance presente en la variable objetivo.

El modelo fue evaluado utilizando un conjunto independiente de prueba.

---

## 📊 Resultados

Los principales resultados obtenidos fueron:

* ROC AUC de **0.84**.
* Lift de **2.89** en el primer decil.
* Identificación efectiva de clientes con mayor probabilidad de abandono.
* El tipo de contrato y la relación entre gasto y permanencia fueron las variables con mayor capacidad predictiva.

---

## 💼 Aplicación en el negocio

Los resultados obtenidos permiten:

* Priorizar campañas de retención.
* Identificar clientes con alto riesgo de abandono.
* Optimizar la asignación de recursos comerciales.
* Diseñar estrategias diferenciadas para cada segmento de clientes.
* Apoyar la toma de decisiones basada en datos.

---

## 📁 Estructura del repositorio

```text
Telco-Churn-Analytics/
│
├── data/
├── notebooks/
├── images/
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar el proyecto

Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/Telco-Churn-Analytics.git
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Abre el notebook ubicado en la carpeta **notebooks** y ejecuta las celdas en orden.

---

## 👨‍💻 Autores

**Jose Luis Garay Ramos**

Estudiante de Ingeniería Estadística e Informática
Universidad Nacional Agraria La Molina

**Diana Chavez**
