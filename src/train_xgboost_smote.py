# =========================================================
# XGBoost Multiclase + SMOTE + Feature Importance
# Dataset: Wine Quality
# =========================================================

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import matplotlib.pyplot as plt


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

# Mapeo clases originales -> codificadas
class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("\nMapping quality -> encoded:")
print(class_mapping)


# =========================================================
# 3. Distribución original
# =========================================================
unique, counts = np.unique(y_encoded, return_counts=True)
print("\nDistribución original:")
print(dict(zip(unique, counts)))


# =========================================================
# 4. SMOTE controlado
#    - Sube: 3, 4, 7, 8
#    - Controla: 5 y 6
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

unique_res, counts_res = np.unique(y_res, return_counts=True)
print("\nDistribución tras SMOTE:")
print(dict(zip(unique_res, counts_res)))


# =========================================================
# 5. Train / Test split (estratificado)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res,
    y_res,
    test_size=0.2,
    random_state=11,
    stratify=y_res
)


# =========================================================
# 6. Modelo XGBoost
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

xgb_model.fit(X_train, y_train)


# =========================================================
# 7. Evaluación
# =========================================================
y_pred = xgb_model.predict(X_test)

print("\nAccuracy XGBoost + SMOTE:",
      accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[str(c) for c in le.inverse_transform(np.unique(y_res))]
))


# =========================================================
# 8. Feature Importance
# =========================================================
importances = xgb_model.get_booster().get_score(importance_type="gain")

feat_imp = (
    pd.DataFrame(importances.items(), columns=["Feature", "Importance"])
      .sort_values(by="Importance", ascending=False)
)

print("\nTop feature importances:")
print(feat_imp.head(10))

# Plot
xgb.plot_importance(
    xgb_model,
    importance_type="gain",
    max_num_features=10
)
plt.tight_layout()
plt.show()


# =========================================================
# 9. Guardar modelo final
# =========================================================
with open("models/xgboost_smote_final.pkl", "wb") as f:
    pickle.dump(
        {
            "model": xgb_model,
            "label_encoder": le
        },
        f
    )

print("\nModelo guardado en models/xgboost_smote_final.pkl")
