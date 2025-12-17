
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,classification_report,ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df = pd.read_csv("winequality-red.csv")

X = df.drop(columns=["quality"])
y = df["quality"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)

class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
inv_class_mapping = {v: k for k, v in class_mapping.items()}


X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=11,stratify=y_encoded)


rf = RandomForestClassifier(random_state=11)

param_grid_rf = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, 30]}

grid_rf = GridSearchCV(rf,param_grid_rf,cv=5,scoring="accuracy",n_jobs=-1,verbose=1)

grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Mejores parámetros:", grid_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

labels = np.unique(y_test)
target_names = [str(le.classes_[i]) for i in labels]


ConfusionMatrixDisplay.from_predictions(y_test,y_pred_rf,display_labels=target_names,xticks_rotation=45)
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()
plt.show()

with open("random_forest_final.pkl", "wb") as f:
    pickle.dump(best_rf, f)


xgb_model = xgb.XGBClassifier(objective="multi:softprob",num_class=len(le.classes_),eval_metric="mlogloss",learning_rate=0.05,max_depth=8,n_estimators=300,random_state=11,n_jobs=-1)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBOOST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))


ConfusionMatrixDisplay.from_predictions(y_test,y_pred_xgb,display_labels=target_names,xticks_rotation=45)
plt.title("Confusion Matrix – XGBoost")
plt.tight_layout()
plt.show()

with open("xgboost_final.pkl", "wb") as f:
    pickle.dump(
        {
            "model": xgb_model,
            "label_encoder": le,
            "class_mapping": class_mapping},f)
