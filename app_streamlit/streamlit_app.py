# streamlit_app.py
# --------------------------------------------------
# Clasificación de Calidad del Vino – Streamlit App
# UI moderna + explicaciones + visualizaciones
# Modelo final: Random Forest (optimizado con GridSearch)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="Dashboard ML – Calidad del Vino",
    page_icon="🍷",
    layout="wide"
)

# -----------------------------
# Título e introducción
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🍷 Clasificación de la Calidad del Vino</h1>
    <p style='text-align: center; font-size:18px;'>
    Modelo final seleccionado: <b>Random Forest</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Barra lateral
# -----------------------------
st.sidebar.header("Navegación")
section = st.sidebar.radio(
    "Ir a",
    [
        "📊 Visión general del dataset",
        "⚖️ Desbalanceo de clases y SMOTE",
        "🤖 Comparación de modelos",
        "🌲 Modelo final: Random Forest",
        "🔮 Realizar una predicción"
    ]
)

# -----------------------------
# Carga de datos y modelo
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

@st.cache_resource
def load_model():
    with open("random_forest_gs.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

X = df.drop(columns=["quality"])
y = df["quality"]

# -----------------------------
# Sección 1 – Visión general del dataset
# -----------------------------
if section == "📊 Visión general del dataset":

    st.subheader("📊 Visión general del dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Primeras filas del dataset**")
        st.dataframe(df.head())

    with col2:
        st.metric("Filas", df.shape[0])
        st.metric("Variables", df.shape[1] - 1)
        st.metric("Variable objetivo", "quality")

    st.divider()

    st.write("**Resumen estadístico**")
    st.dataframe(df.describe())

    st.divider()

    st.write("**Mapa de calor de correlaciones**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Sección 2 – Desbalanceo de clases
# -----------------------------
elif section == "⚖️ Desbalanceo de clases y SMOTE":

    st.subheader("⚖️ Desbalanceo de clases y SMOTE")

    st.write(
        """
        La variable objetivo **quality** está claramente desbalanceada.
        La mayoría de los vinos tienen puntuaciones **5–6**, mientras que
        los valores extremos (3, 4, 7 y 8) están infrarepresentados.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        y.value_counts().sort_index().plot(
            kind="bar",
            ax=ax
        )
        ax.set_title("Distribución original de clases")
        ax.set_xlabel("Calidad")
        ax.set_ylabel("Número de muestras")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        y.value_counts().sort_index().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Distribución porcentual de clases")
        st.pyplot(fig)

    st.info(
        "Se probó SMOTE para reforzar las clases minoritarias, pero no mejoró la accuracy global en comparación con Random Forest sin re-muestreo."
    )

# -----------------------------
# Sección 3 – Comparación de modelos
# -----------------------------
elif section == "🤖 Comparación de modelos":

    st.subheader("🤖 Comparación de modelos")

    models = [
        "Regresión Logística",
        "AdaBoost",
        "Gradient Boosting",
        "Random Forest",
        "XGBoost",
        "XGBoost + SMOTE"
    ]

    accuracies = [0.56, 0.55, 0.64, 0.68, 0.66, 0.66]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparativa de accuracy entre modelos")
    ax.set_xticklabels(models, rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    st.success(
        "**Random Forest** alcanzó el mejor equilibrio entre rendimiento y estabilidad, por lo que se selecciona como modelo final."
    )

# -----------------------------
# Sección 4 – Modelo final
# -----------------------------
elif section == "🌲 Modelo final: Random Forest":

    st.subheader("🌲 Modelo final – Random Forest")

    st.write(
        """
        **Mejores hiperparámetros (GridSearchCV)**
        - n_estimators: 200
        - max_depth: 20
        - Accuracy en test: **0.68**
        """
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=11
    )

    y_pred = model.predict(X_test)

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        ax.set_title("Matriz de confusión")
        st.pyplot(fig)

    with col2:
        st.write("**Informe de clasificación**")
        st.text(classification_report(y_test, y_pred))

# -----------------------------
# Sección 5 – Predicción
# -----------------------------
elif section == "🔮 Realizar una predicción":

    st.subheader("🔮 Predicción de la calidad del vino")


    st.write("Ajusta las propiedades químicas y predice la calidad del vino.")


    input_data = {}


    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            input_data[col] = st.number_input(
                col,
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )


    input_df = pd.DataFrame([input_data])


    if st.button("Predecir calidad 🍷"):
        prediction = model.predict(input_df)[0]
        st.success(f"Calidad de vino predicha: **{prediction}**")


# -----------------------------
# Pie de página
# -----------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:14px;'>Proyecto de Machine Learning • Dashboard en Streamlit</p>",
    unsafe_allow_html=True
)
