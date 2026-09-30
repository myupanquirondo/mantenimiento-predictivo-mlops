import platform

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

# ============================================================
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance"
REGISTERED_MODEL_NAME = "predictive-maintenance-xgboost"

# Configuración seleccionada mediante Stratified 5-Fold CV
N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.03
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
SCALE_POS_WEIGHT = 25

# Umbral operativo seleccionado mediante CV
DECISION_THRESHOLD = 0.35

# Métricas obtenidas durante 5-Fold CV
CV_PRECISION_MEAN = 0.2456
CV_PRECISION_STD = 0.0058

CV_RECALL_MEAN = 0.9558
CV_RECALL_STD = 0.0092
CV_RECALL_MIN = 0.9412
CV_RECALL_MAX = 0.9701

CV_F1_MEAN = 0.3907
CV_F1_STD = 0.0075

CV_TP = 324
CV_FN = 15
CV_FP = 996
CV_TN = 8665


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


print(f"Registros utilizados para entrenamiento final: {len(X)}")
print(f"Fallas: {int(y.sum())}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

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


# ============================================================
# MODELO DE ALTA SENSIBILIDAD
# ============================================================

classifier = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    scale_pos_weight=SCALE_POS_WEIGHT,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# ============================================================
# ENTRENAMIENTO FINAL
# ============================================================

print("\nEntrenando modelo de alta sensibilidad...")

final_model.fit(X, y)


# ============================================================
# REGISTRO EN MLFLOW
# ============================================================

print("\nRegistrando modelo en MLflow...")

with mlflow.start_run(
    run_name="xgboost-high-recall-final"
) as run:
    # --------------------------------------------------------
    # HIPERPARÁMETROS
    # --------------------------------------------------------

    mlflow.log_params(
        {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "decision_threshold": DECISION_THRESHOLD,
            "selection_strategy": (
                "recall>=0.95_then_maximize_f1"
            ),
            "validation_strategy": (
                "StratifiedKFold_5"
            ),
            "training_records": len(X),
        }
    )

    # --------------------------------------------------------
    # MÉTRICAS DE CROSS-VALIDATION
    # --------------------------------------------------------

    mlflow.log_metrics(
        {
            "cv_precision_mean": CV_PRECISION_MEAN,
            "cv_precision_std": CV_PRECISION_STD,
            "cv_recall_mean": CV_RECALL_MEAN,
            "cv_recall_std": CV_RECALL_STD,
            "cv_recall_min": CV_RECALL_MIN,
            "cv_recall_max": CV_RECALL_MAX,
            "cv_f1_mean": CV_F1_MEAN,
            "cv_f1_std": CV_F1_STD,
            "cv_tp": CV_TP,
            "cv_fn": CV_FN,
            "cv_fp": CV_FP,
            "cv_tn": CV_TN,
        }
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    mlflow.set_tags(
        {
            "stage": "final-high-recall",
            "model_type": "XGBoost",
            "model_profile": "high-recall",
            "primary_metric": "recall",
            "decision_threshold": str(
                DECISION_THRESHOLD
            ),
            "cv_folds": "5",
            "evaluation_type": (
                "out-of-fold cross-validation"
            ),
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "xgboost_version": xgboost.__version__,
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
    )

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        input_example=X.head(5),
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )

    print(f"\nRun ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")


# ============================================================
# RESULTADO
# ============================================================

print("\nModelo de alta sensibilidad registrado correctamente.")

print(
    f"Nombre del modelo: "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Threshold operativo: "
    f"{DECISION_THRESHOLD}"
)

print(
    f"Recall CV: "
    f"{CV_RECALL_MEAN:.4f} "
    f"± {CV_RECALL_STD:.4f}"
)

print(
    f"Precision CV: "
    f"{CV_PRECISION_MEAN:.4f} "
    f"± {CV_PRECISION_STD:.4f}"
)

print(
    f"F1 CV: "
    f"{CV_F1_MEAN:.4f} "
    f"± {CV_F1_STD:.4f}"
)