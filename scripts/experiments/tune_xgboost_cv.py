import itertools

import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

# ============================================================
# CONFIGURACIÓN
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance"

N_SPLITS = 5
RANDOM_STATE = 42

# Objetivo mínimo que queremos conseguir de forma promedio
# entre los 5 folds.
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
# CONFIGURACIONES A PROBAR
# ============================================================

n_estimators_values = [200, 300]
max_depth_values = [3, 5, 7]
learning_rate_values = [0.03, 0.05]
scale_pos_weight_values = [5, 10, 15, 20, 28]

thresholds = np.arange(0.05, 0.51, 0.05)

configurations = list(
    itertools.product(
        n_estimators_values,
        max_depth_values,
        learning_rate_values,
        scale_pos_weight_values,
    )
)

print(f"Registros totales: {len(X)}")
print(f"Fallas totales: {int(y.sum())}")

print(f"\nConfiguraciones XGBoost: {len(configurations)}")
print(f"Umbrales por configuración: {len(thresholds)}")
print(f"Folds: {N_SPLITS}")

print(
    "Evaluaciones modelo + threshold + fold: "
    f"{len(configurations) * len(thresholds) * N_SPLITS}"
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
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
        random_state=RANDOM_STATE,
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

print("\nIniciando Stratified 5-Fold Cross-Validation...\n")


for config_index, configuration in enumerate(
    configurations,
    start=1,
):
    (
        n_estimators,
        max_depth,
        learning_rate,
        scale_pos_weight,
    ) = configuration

    # Guardamos las probabilidades y etiquetas de cada fold.
    # Así podemos evaluar exactamente el mismo threshold
    # sobre todas las predicciones out-of-fold.
    fold_probabilities = []
    fold_targets = []

    for train_index, val_index in cv.split(X, y):
        X_fold_train = X.iloc[train_index]
        X_fold_val = X.iloc[val_index]

        y_fold_train = y.iloc[train_index]
        y_fold_val = y.iloc[val_index]

        model = create_pipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        probabilities = model.predict_proba(
            X_fold_val
        )[:, 1]

        fold_probabilities.append(probabilities)
        fold_targets.append(y_fold_val.to_numpy())

    # ========================================================
    # EVALUACIÓN DE CADA THRESHOLD
    # ========================================================

    for threshold in thresholds:
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        total_tp = 0
        total_fn = 0
        total_fp = 0

        for y_fold_val, probabilities in zip(
            fold_targets,
            fold_probabilities,
            strict=True,
        ):
            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_fold_val,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_fold_val,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_fold_val,
                predictions,
                zero_division=0,
            )

            tp = int(
                (
                    (predictions == 1)
                    & (y_fold_val == 1)
                ).sum()
            )

            fn = int(
                (
                    (predictions == 0)
                    & (y_fold_val == 1)
                ).sum()
            )

            fp = int(
                (
                    (predictions == 1)
                    & (y_fold_val == 0)
                ).sum()
            )

            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1_scores.append(f1)

            total_tp += tp
            total_fn += fn
            total_fp += fp

        results.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "scale_pos_weight": scale_pos_weight,
                "threshold": float(threshold),
                "precision_mean": float(
                    np.mean(fold_precisions)
                ),
                "precision_std": float(
                    np.std(fold_precisions)
                ),
                "recall_mean": float(
                    np.mean(fold_recalls)
                ),
                "recall_std": float(
                    np.std(fold_recalls)
                ),
                "recall_min": float(
                    np.min(fold_recalls)
                ),
                "f1_mean": float(
                    np.mean(fold_f1_scores)
                ),
                "f1_std": float(
                    np.std(fold_f1_scores)
                ),
                "tp": total_tp,
                "fn": total_fn,
                "fp": total_fp,
            }
        )

    print(
        f"[{config_index}/{len(configurations)}] "
        f"trees={n_estimators}, "
        f"depth={max_depth}, "
        f"lr={learning_rate}, "
        f"weight={scale_pos_weight}"
    )


# ============================================================
# FILTRAR POR RECALL OBJETIVO
# ============================================================

candidates = [
    result
    for result in results
    if result["recall_mean"] >= TARGET_RECALL
]


if candidates:
    # Priorizamos:
    # 1. Recall promedio >= 90 %
    # 2. Mayor F1 promedio
    # 3. Mayor Precision promedio
    # 4. Menor variación de Recall
    best_result = max(
        candidates,
        key=lambda result: (
            result["f1_mean"],
            result["precision_mean"],
            -result["recall_std"],
        ),
    )

else:
    print(
        f"\nNinguna configuración alcanzó "
        f"Recall promedio >= {TARGET_RECALL:.0%}."
    )

    best_result = max(
        results,
        key=lambda result: (
            result["recall_mean"],
            result["f1_mean"],
        ),
    )


# ============================================================
# MEJOR CONFIGURACIÓN
# ============================================================

print("\n" + "=" * 65)
print("MEJOR CONFIGURACIÓN EN 5-FOLD CROSS-VALIDATION")
print("=" * 65)

print(f"n_estimators: {best_result['n_estimators']}")
print(f"max_depth: {best_result['max_depth']}")
print(f"learning_rate: {best_result['learning_rate']}")
print(
    "scale_pos_weight: "
    f"{best_result['scale_pos_weight']}"
)
print(f"threshold: {best_result['threshold']:.2f}")

print("\nMétricas promedio:")

print(
    f"Precision: "
    f"{best_result['precision_mean']:.4f} "
    f"± {best_result['precision_std']:.4f}"
)

print(
    f"Recall: "
    f"{best_result['recall_mean']:.4f} "
    f"± {best_result['recall_std']:.4f}"
)

print(
    f"Recall mínimo de un fold: "
    f"{best_result['recall_min']:.4f}"
)

print(
    f"F1: "
    f"{best_result['f1_mean']:.4f} "
    f"± {best_result['f1_std']:.4f}"
)

print("\nPredicciones out-of-fold acumuladas:")

print(f"TP: {best_result['tp']}")
print(f"FN: {best_result['fn']}")
print(f"FP: {best_result['fp']}")

print(
    "Fallas detectadas: "
    f"{best_result['tp']}/"
    f"{best_result['tp'] + best_result['fn']}"
)


# ============================================================
# TOP 10
# ============================================================

top_candidates = sorted(
    candidates,
    key=lambda result: (
        result["f1_mean"],
        result["precision_mean"],
        -result["recall_std"],
    ),
    reverse=True,
)[:10]


if top_candidates:
    print("\n" + "=" * 65)
    print("TOP 10 CON RECALL PROMEDIO >= 90%")
    print("=" * 65)

    for position, result in enumerate(
        top_candidates,
        start=1,
    ):
        print(
            f"{position:>2}. "
            f"trees={result['n_estimators']}, "
            f"depth={result['max_depth']}, "
            f"lr={result['learning_rate']}, "
            f"weight={result['scale_pos_weight']}, "
            f"threshold={result['threshold']:.2f} | "
            f"P={result['precision_mean']:.4f} | "
            f"R={result['recall_mean']:.4f} "
            f"±{result['recall_std']:.4f} | "
            f"Rmin={result['recall_min']:.4f} | "
            f"F1={result['f1_mean']:.4f}"
        )


# ============================================================
# MLFLOW
# ============================================================

with mlflow.start_run(
    run_name="xgboost-stratified-5fold-cv"
):
    mlflow.log_params(
        {
            "n_splits": N_SPLITS,
            "n_estimators": best_result[
                "n_estimators"
            ],
            "max_depth": best_result["max_depth"],
            "learning_rate": best_result[
                "learning_rate"
            ],
            "scale_pos_weight": best_result[
                "scale_pos_weight"
            ],
            "threshold": best_result["threshold"],
            "target_recall": TARGET_RECALL,
        }
    )

    mlflow.log_metrics(
        {
            "cv_precision_mean": best_result[
                "precision_mean"
            ],
            "cv_precision_std": best_result[
                "precision_std"
            ],
            "cv_recall_mean": best_result[
                "recall_mean"
            ],
            "cv_recall_std": best_result[
                "recall_std"
            ],
            "cv_recall_min": best_result[
                "recall_min"
            ],
            "cv_f1_mean": best_result[
                "f1_mean"
            ],
            "cv_f1_std": best_result[
                "f1_std"
            ],
            "cv_tp": best_result["tp"],
            "cv_fn": best_result["fn"],
            "cv_fp": best_result["fp"],
        }
    )

    mlflow.set_tags(
        {
            "stage": "cross-validation",
            "validation_strategy": (
                "StratifiedKFold-5"
            ),
            "selection_metric": (
                "recall>=0.90 then best F1"
            ),
        }
    )


print("\nCross-validation terminada.")
print("No se utilizó el resultado del TEST anterior para seleccionar.")