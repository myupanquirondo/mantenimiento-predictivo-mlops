import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
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
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance"

# Mejor scale_pos_weight obtenido anteriormente
BEST_SCALE_POS_WEIGHT = 10

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# CARGA DE DATOS
# ============================================================

dataset = fetch_ucirepo(id=601)
df = dataset.data.original


# ============================================================
# VARIABLES
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
# PIPELINE
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

classifier = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=BEST_SCALE_POS_WEIGHT,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("Entrenando modelo para optimización del umbral...")

model.fit(X_train_inner, y_train_inner)


# ============================================================
# PROBABILIDADES EN VALIDACIÓN
# ============================================================

y_val_prob = model.predict_proba(X_val)[:, 1]


# ============================================================
# UMBRALES A EVALUAR
# ============================================================

thresholds = np.arange(0.50, 0.04, -0.05)


# ============================================================
# EVALUACIÓN DE UMBRALES
# ============================================================

results = []

print("\nResultados por umbral:\n")

print(
    f"{'Umbral':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Fallas detectadas':<20}"
)

print("-" * 66)

total_failures = int(y_val.sum())

for threshold in thresholds:
    y_pred = (y_val_prob >= threshold).astype(int)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    detected_failures = int(
        ((y_pred == 1) & (y_val.to_numpy() == 1)).sum()
    )

    results.append(
        {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detected_failures": detected_failures,
        }
    )

    print(
        f"{threshold:<10.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{detected_failures}/{total_failures}"
    )


# ============================================================
# MEJOR UMBRAL CON RECALL >= 90 %
# ============================================================

target_recall = 0.90

candidates = [
    result
    for result in results
    if result["recall"] >= target_recall
]

if candidates:
    # Entre los que alcanzan 90 % de Recall,
    # elegimos el que tenga mayor F1.
    best_result = max(
        candidates,
        key=lambda result: result["f1"],
    )

    print("\nMejor umbral con Recall >= 90%:")
    print(f"Umbral: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1: {best_result['f1']:.4f}")
    print(
        "Fallas detectadas: "
        f"{best_result['detected_failures']}/{total_failures}"
    )
else:
    best_result = max(
        results,
        key=lambda result: result["recall"],
    )

    print("\nNingún umbral alcanzó Recall >= 90%.")
    print("Mayor Recall encontrado:")
    print(f"Umbral: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1: {best_result['f1']:.4f}")


# ============================================================
# REGISTRO DEL EXPERIMENTO EN MLFLOW
# ============================================================

with mlflow.start_run(run_name="threshold-optimization"):
    mlflow.log_param(
        "scale_pos_weight",
        BEST_SCALE_POS_WEIGHT,
    )

    mlflow.log_param(
        "target_recall",
        target_recall,
    )

    mlflow.log_param(
        "selected_threshold",
        best_result["threshold"],
    )

    mlflow.log_metrics(
        {
            "selected_precision": best_result["precision"],
            "selected_recall": best_result["recall"],
            "selected_f1": best_result["f1"],
        }
    )

    mlflow.set_tag(
        "stage",
        "threshold-optimization",
    )


print("\nOptimización de umbral terminada.")