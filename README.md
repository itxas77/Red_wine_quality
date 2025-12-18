# 🍷 Clasificación de la Calidad del Vino – Proyecto de Machine Learning

## 📌 Descripción del proyecto

Este proyecto presenta una **solución completa de Machine Learning (end‑to‑end)** para predecir la **calidad del vino tinto** a partir de sus propiedades fisicoquímicas. El objetivo es construir, evaluar y desplegar un modelo de clasificación robusto y comunicar los resultados mediante un **dashboard interactivo en Streamlit**.

El proyecto está diseñado tanto como **ejercicio técnico de ML** como **proyecto de portfolio**, adecuado para presentaciones, defensa de bootcamp y entrevistas técnicas.

---

## 🎯 Objetivos

* Analizar y comprender el dataset de Wine Quality (vino tinto)
* Tratar el **desbalanceo de clases** y evaluar técnicas de remuestreo
* Entrenar y comparar distintos modelos de clasificación
* Optimizar el mejor modelo mediante **GridSearchCV**
* Interpretar el rendimiento usando métricas adecuadas
* Desplegar el modelo final en una **aplicación interactiva con Streamlit**

---

## 📊 Dataset

* **Fuente:** UCI Machine Learning Repository – Wine Quality Dataset (Red Wine)
* **Muestras:** 1.599 vinos
* **Variables:** 11 características fisicoquímicas
* **Variable objetivo:** `quality` (valores enteros de 3 a 8)

### Variables de entrada

* fixed acidity
* volatile acidity
* citric acid
* residual sugar
* chlorides
* free sulfur dioxide
* total sulfur dioxide
* density
* pH
* sulphates
* alcohol

---

## ⚖️ Desbalanceo de clases

La variable objetivo está **claramente desbalanceada**, concentrándose la mayoría de observaciones en las calidades **5 y 6**.

* Clases minoritarias: 3, 4, 7 y 8
* Se evaluó el uso de **SMOTE** para equilibrar las clases
* **Conclusión:** SMOTE no mejoró la accuracy global, por lo que no se utilizó en el modelo final

---

## 🤖 Modelos evaluados

Se entrenaron y compararon los siguientes modelos:

* Regresión Logística
* AdaBoost
* Gradient Boosting
* Random Forest
* XGBoost
* XGBoost + SMOTE

### Comparación de modelos (Accuracy)

| Modelo              | Accuracy |
| ------------------- | -------- |
| Regresión Logística | 0.56     |
| AdaBoost            | 0.55     |
| Gradient Boosting   | 0.64     |
| **Random Forest**   | **0.68** |
| XGBoost             | 0.66     |
| XGBoost + SMOTE     | 0.66     |

---

## 🌲 Modelo final – Random Forest

El **Random Forest Classifier** fue seleccionado por ofrecer el mejor equilibrio entre rendimiento y estabilidad.

### Mejores hiperparámetros (GridSearchCV)

* `n_estimators`: 200
* `max_depth`: 20
* Accuracy en test: **0.68**

### Métricas de evaluación

* Accuracy (train y test)
* Matriz de confusión
* Classification report (precisión, recall y F1‑score)

---

## 🔮 Aplicación Streamlit

Se desarrolló un **dashboard interactivo en Streamlit** para presentar los resultados y permitir predicciones en tiempo real.

### Funcionalidades principales

* Visión general del dataset
* Estadísticas descriptivas
* Mapa de calor de correlaciones
* Análisis de desbalanceo de clases
* Comparativa de modelos
* Matriz de confusión y classification report
* **Predicción interactiva de la calidad del vino**
* Feedback visual mediante **imágenes asociadas a cada nivel de calidad**

---

## 🖼️ Mejora visual

Cada valor de calidad predicho se asocia a una imagen representativa:

* Baja calidad → imágenes más oscuras y menos atractivas
* Alta calidad → imágenes premium y elegantes

Esto mejora la interpretabilidad y hace la aplicación más intuitiva para usuarios no técnicos.

---

## 🛠️ Tecnologías utilizadas

* Python 3
* pandas, numpy
* scikit‑learn
* XGBoost
* matplotlib, seaborn
* Streamlit
* pickle

---

## 🚀 Cómo ejecutar el proyecto

### 1️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar la aplicación Streamlit

```bash
python -m streamlit run streamlit_app.py
```

---

## 📁 Estructura del proyecto

```
Red_wine_quality/
│── streamlit_app.py
│── winequality-red.csv
│── random_forest_gs.pkl
│── images/
│   ├── wine_banner.jpg
│   ├── quality_3.jpg
│   ├── quality_4.jpg
│   ├── quality_5.jpg
│   ├── quality_6.jpg
│   ├── quality_7.jpg
│   └── quality_8.jpg
│── README.md
```

---

## 🧠 Conclusiones clave

* Los modelos ensemble superan a los modelos lineales en este dataset
* El desbalanceo de clases debe evaluarse más allá de la accuracy
* Los dashboards interactivos mejoran la comunicación de resultados
* Random Forest ofrece un excelente equilibrio entre rendimiento e interpretabilidad

---

## 👩‍💻 Autora

**Itxaso Campos Molina**
Interés en Data Science y Machine Learning
📧 Email: [itxas.77@gmail.com](mailto:itxas.77@gmail.com)

---

## 📌 Posibles mejoras futuras

* Enfoque de regresión para predecir calidad continua
* Interpretabilidad con valores SHAP
* Calibración del modelo y aprendizaje sensible al coste
* Despliegue en Streamlit Cloud u otra plataforma

---

🍷 *Este proyecto combina ciencia de datos, machine learning y comunicación visual para transformar datos en conocimiento accionable.*
