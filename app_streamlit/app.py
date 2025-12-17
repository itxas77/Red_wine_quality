import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Random Forest – Feature Importance", layout="wide")

st.title("🌲 Random Forest – Feature Importance")

# --------------------------------------------------
# 1. Subida de datos
# --------------------------------------------------
uploaded_file = st.file_uploader("data/winequality-red.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Selección de target
    target = st.selectbox("quality", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # --------------------------------------------------
        # 2. Train / Test split
        # --------------------------------------------------
        test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=11, stratify=y
        )

        # --------------------------------------------------
        # 3. Entrenamiento
        # --------------------------------------------------
        if st.button("Entrenar modelo"):
            with st.spinner("Entrenando Random Forest..."):
                rf = RandomForestClassifier(random_state=11)

                param_grid = {
                    "n_estimators": [200, 300],
                    "max_depth": [10, 20, 30]
                }

                grid = GridSearchCV(
                    rf,
                    param_grid,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)
                best_rf = grid.best_estimator_

                # --------------------------------------------------
                # 4. Evaluación
                # --------------------------------------------------
                y_pred = best_rf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"Accuracy (test): {acc:.3f}")
                st.write("Mejores parámetros:", grid.best_params_)

                # --------------------------------------------------
                # 5. Feature Importance
                # --------------------------------------------------
                importances = best_rf.feature_importances_

                feature_importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values(by="importance", ascending=False)

                st.subheader("Feature Importance")
                st.dataframe(feature_importance)

                # --------------------------------------------------
                # 6. Guardar modelo
                # --------------------------------------------------
                with open("random_forest_gs.pkl", "wb") as f:
                    pickle.dump(best_rf, f)

                st.info("Modelo guardado como random_forest_gs.pkl")
