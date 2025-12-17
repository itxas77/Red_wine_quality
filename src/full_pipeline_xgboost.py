# =========================================================
# Pipeline completo: Entrenamiento XGBoost + SMOTE + Predicción
# =========================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

import xgboost as xgb

# =========================================================
# 1. Carga de datos
# =========================================================
df = pd.read_csv("winequality-red.csv")

X = df.drop(columns=["quality"])
y = df["quality"]

# =========================================================
# 2. Label Encoding
# =========================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Mapping quality -> encoded:", class_mapping)

# =========================================================
# 3. SMOTE controlado
# =========================================================
sampling_strategy = {
    class_mapping[3]: 800,
    class_mapping[4]: 800,
    class_mapping[7]: 800,
    class_mapping[8]: 800,
    class_mapping[5]: 1000,
    class_mapping[6]: 1000
}

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5
)

X_res, y_res = smote.fit_resample(X, y_encoded)
print("Distribución tras SMOTE:", dict(zip(*np.unique(y_res, return_counts=True))))

# =========================================================
# 4. Train / Test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=11, stratify=y_res
)

# =========================================================
# 5. XGBoost - Base
# =========================================================
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_res)),
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)

# =========================================================
# 6. GridSearch opcional (descomentar si quieres tunear)
# =========================================================
# param_grid = {
#     "n_estimators": [200, 300, 500],
#     "learning_rate": [0.05, 0.1, 0.2],
#     "max_depth": [3, 4, 6, 8],
#     "subsample": [0.8, 0.9, 1.0],
#     "colsample_bytree": [0.7, 0.8, 1.0]
# }
# grid = GridSearchCV(
#     xgb_model,
#     param_grid,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=-1,
#     verbose=1
# )
# grid.fit(X_train, y_train)
# xgb_model = grid.best_estimator_

# =========================================================
# 7. Entrenamiento
# =========================================================
xgb_model.fit(X_train, y_train)

# =========================================================
# 8. Evaluación
# =========================================================
y_pred = xgb_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[str(c) for c in le.inverse_transform(np.unique(y_res))]
))

# =========================================================
# 9. Feature importance
# =========================================================
importances = xgb_model.get_booster().get_score(importance_type="gain")
feat_imp = pd.DataFrame(importances.items(), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
print("\nTop feature importances:")
print(feat_imp.head(10))

xgb.plot_importance(xgb_model, importance_type="gain", max_num_features=10)
plt.tight_layout()
plt.show()

# =========================================================
# 10. Guardar modelo final
# =========================================================
with open("models/xgboost_pipeline_final.pkl", "wb") as f:
    pickle.dump(
        {"model": xgb_model, "label_encoder": le},
        f
    )
print("\nModelo guardado en models/xgboost_pipeline_final.pkl")

# =========================================================
# 11. Predicción nuevos vinos
# =========================================================
# Ajusta la ruta a tu CSV de nuevos vinos
new_data_path = "nuevos_vinos.csv"
df_new = pd.read_csv(new_data_path)
X_new = df_new  # todas las columnas features

y_pred_new_encoded = xgb_model.predict(X_new)
y_prob_new = xgb_model.predict_proba(X_new)
y_pred_new = le.inverse_transform(y_pred_new_encoded)

# DataFrame de resultados
prob_df = pd.DataFrame(
    y_prob_new,
    columns=[f"prob_quality_{cls}" for cls in le.inverse_transform(np.arange(y_prob_new.shape[1]))]
)
results_df = pd.concat(
    [df_new.reset_index(drop=True), pd.Series(y_pred_new, name="pred_quality"), prob_df],
    axis=1
)

# Guardar resultados
output_path = "predicciones_nuevos_vinos.csv"
results_df.to_csv(output_path, index=False)
print(f"\nPredicciones guardadas en {output_path}")
print(results_df.head())
