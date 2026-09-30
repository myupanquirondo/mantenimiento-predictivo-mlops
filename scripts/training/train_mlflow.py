import platform

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

# ============================================================
# CONFIGURACIÓN DE MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance"
REGISTERED_MODEL_NAME = "predictive-maintenance-xgboost"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# CARGA DE DATOS
# ============================================================

dataset = fetch_ucirepo(id=601)

df = dataset.data.original


# ============================================================
# SELECCIÓN DE VARIABLES
# ============================================================

features = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

target = "Machine failure"

categorical_features = ["Type"]

X = df[features].copy()
y = df[target].copy()


# ============================================================
# DIVISIÓN DE DATOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train,
)


# ============================================================
# FUNCIÓN PARA CREAR EL PIPELINE
# ============================================================

def create_pipeline(scale_pos_weight):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ============================================================
# CONFIGURACIONES DE XGBOOST
# ============================================================

experiments = [
    {
        "run_name": "xgboost-weight-1",
        "scale_pos_weight": 1,
    },
    {
        "run_name": "xgboost-weight-2",
        "scale_pos_weight": 2,
    },
    {
        "run_name": "xgboost-weight-5",
        "scale_pos_weight": 5,
    },
    {
        "run_name": "xgboost-weight-10",
        "scale_pos_weight": 10,
    },
    {
        "run_name": "xgboost-weight-15",
        "scale_pos_weight": 15,
    },
]


# ============================================================
# EXPERIMENTACIÓN Y SELECCIÓN DEL MEJOR MODELO
# ============================================================

best_f1 = -1.0
best_run_name = None
best_scale_pos_weight = None

for experiment in experiments:
    run_name = experiment["run_name"]
    scale_pos_weight = experiment["scale_pos_weight"]

    with mlflow.start_run(run_name=run_name):
        model = create_pipeline(scale_pos_weight)

        model.fit(X_train_inner, y_train_inner)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        pr_auc = average_precision_score(y_val, y_prob)

        mlflow.log_params(
            {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
            }
        )

        mlflow.set_tags(
            {
                "stage": "validation",
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "xgboost_version": xgboost.__version__,
                "scikit_learn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "pr_auc": pr_auc,
            }
        )

        print(
            f"{run_name} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"PR-AUC: {pr_auc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_run_name = run_name
            best_scale_pos_weight = scale_pos_weight


# ============================================================
# RESULTADO DE LA SELECCIÓN
# ============================================================

print("\nMejor configuración encontrada:")
print(f"Run: {best_run_name}")
print(f"scale_pos_weight: {best_scale_pos_weight}")
print(f"F1 validación: {best_f1:.4f}")


# ============================================================
# ENTRENAMIENTO DEL MODELO FINAL
# ============================================================

final_model = create_pipeline(best_scale_pos_weight)

# Ahora usamos todo el conjunto de entrenamiento.
final_model.fit(X_train, y_train)


# ============================================================
# EVALUACIÓN FINAL EN TEST
# ============================================================

y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_pr_auc = average_precision_score(y_test, y_test_prob)

print("\nResultados finales en TEST:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"PR-AUC: {test_pr_auc:.4f}")


# ============================================================
# REGISTRO DEL MODELO FINAL EN MLFLOW
# ============================================================

with mlflow.start_run(run_name="xgboost-final-model"):
    mlflow.log_params(
        {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": best_scale_pos_weight,
            "selection_metric": "f1_score",
        }
    )

    mlflow.log_metrics(
        {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1,
            "test_pr_auc": test_pr_auc,
        }
    )

    mlflow.set_tags(
        {
            "stage": "final",
            "selected_from": best_run_name,
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "xgboost_version": xgboost.__version__,
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
    )

    mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        input_example=X_train.head(5),
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )


print("\nModelo final registrado correctamente en MLflow.")
print(f"Nombre del modelo: {REGISTERED_MODEL_NAME}")