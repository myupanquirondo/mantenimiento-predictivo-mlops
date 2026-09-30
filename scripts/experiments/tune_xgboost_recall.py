import itertools

import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
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

# Queremos detectar al menos el 90 % de las fallas reales.
TARGET_RECALL = 0.90

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

# TEST queda reservado.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# TRAIN se divide nuevamente en entrenamiento y validación.
X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train,
)

print(f"Train: {X_train_inner.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test reservado: {X_test.shape}")


# ============================================================
# CONFIGURACIONES A PROBAR
# ============================================================

n_estimators_values = [200, 300]
max_depth_values = [3, 5, 7]
learning_rate_values = [0.03, 0.05]
scale_pos_weight_values = [5, 10, 15, 20, 28]

# Umbrales más detallados que en el experimento anterior.
thresholds = np.arange(0.05, 0.51, 0.05)

configurations = list(
    itertools.product(
        n_estimators_values,
        max_depth_values,
        learning_rate_values,
        scale_pos_weight_values,
    )
)

print(f"\nConfiguraciones XGBoost: {len(configurations)}")
print(f"Umbrales por configuración: {len(thresholds)}")
print(
    "Combinaciones modelo + umbral: "
    f"{len(configurations) * len(thresholds)}"
)


# ============================================================
# FUNCIÓN PARA CREAR PIPELINE
# ============================================================


def create_pipeline(
    n_estimators,
    max_depth,
    learning_rate,
    scale_pos_weight,
):
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
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
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
# BÚSQUEDA
# ============================================================

results = []

total_failures = int(y_val.sum())

print(f"\nFallas reales en VALIDATION: {total_failures}")
print("\nIniciando búsqueda...\n")


for index, configuration in enumerate(configurations, start=1):
    (
        n_estimators,
        max_depth,
        learning_rate,
        scale_pos_weight,
    ) = configuration

    model = create_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train_inner, y_train_inner)

    y_probability = model.predict_proba(X_val)[:, 1]

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        precision = precision_score(
            y_val,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
        ).ravel()

        results.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "scale_pos_weight": scale_pos_weight,
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    print(
        f"[{index}/{len(configurations)}] "
        f"trees={n_estimators}, "
        f"depth={max_depth}, "
        f"lr={learning_rate}, "
        f"weight={scale_pos_weight}"
    )


# ============================================================
# FILTRAR MODELOS CON RECALL >= OBJETIVO
# ============================================================

candidates = [
    result
    for result in results
    if result["recall"] >= TARGET_RECALL
]


if not candidates:
    print(
        f"\nNinguna configuración alcanzó "
        f"Recall >= {TARGET_RECALL:.0%}."
    )

    best_result = max(
        results,
        key=lambda result: (
            result["recall"],
            result["precision"],
        ),
    )

else:
    # Entre todos los modelos que alcanzan el Recall objetivo,
    # buscamos primero el mayor F1 y luego la mayor Precision.
    best_result = max(
        candidates,
        key=lambda result: (
            result["f1"],
            result["precision"],
        ),
    )


# ============================================================
# MOSTRAR MEJOR RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("MEJOR CONFIGURACIÓN EN VALIDATION")
print("=" * 60)

print(f"n_estimators: {best_result['n_estimators']}")
print(f"max_depth: {best_result['max_depth']}")
print(f"learning_rate: {best_result['learning_rate']}")
print(f"scale_pos_weight: {best_result['scale_pos_weight']}")
print(f"threshold: {best_result['threshold']:.2f}")

print("\nMétricas:")

print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"F1: {best_result['f1']:.4f}")

print("\nMatriz de confusión:")

print(f"TP: {best_result['tp']}")
print(f"FN: {best_result['fn']}")
print(f"FP: {best_result['fp']}")
print(f"TN: {best_result['tn']}")

print(
    "\nFallas detectadas: "
    f"{best_result['tp']}/{total_failures}"
)


# ============================================================
# TOP 10
# ============================================================

top_candidates = sorted(
    candidates,
    key=lambda result: (
        result["f1"],
        result["precision"],
    ),
    reverse=True,
)[:10]


if top_candidates:
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURACIONES CON RECALL >= 90%")
    print("=" * 60)

    for position, result in enumerate(top_candidates, start=1):
        print(
            f"{position:>2}. "
            f"trees={result['n_estimators']}, "
            f"depth={result['max_depth']}, "
            f"lr={result['learning_rate']}, "
            f"weight={result['scale_pos_weight']}, "
            f"threshold={result['threshold']:.2f} | "
            f"P={result['precision']:.4f} | "
            f"R={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"TP={result['tp']} | "
            f"FN={result['fn']} | "
            f"FP={result['fp']}"
        )


# ============================================================
# REGISTRAR MEJOR RESULTADO EN MLFLOW
# ============================================================

with mlflow.start_run(run_name="xgboost-recall-tuning"):
    mlflow.log_params(
        {
            "n_estimators": best_result["n_estimators"],
            "max_depth": best_result["max_depth"],
            "learning_rate": best_result["learning_rate"],
            "scale_pos_weight": best_result["scale_pos_weight"],
            "threshold": best_result["threshold"],
            "target_recall": TARGET_RECALL,
        }
    )

    mlflow.log_metrics(
        {
            "validation_precision": best_result["precision"],
            "validation_recall": best_result["recall"],
            "validation_f1": best_result["f1"],
            "validation_tp": best_result["tp"],
            "validation_fn": best_result["fn"],
            "validation_fp": best_result["fp"],
            "validation_tn": best_result["tn"],
        }
    )

    mlflow.set_tag(
        "stage",
        "recall-tuning",
    )


print("\nBúsqueda terminada.")
print("TEST todavía NO fue utilizado.")