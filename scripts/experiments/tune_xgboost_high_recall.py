import itertools

import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
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

# En este experimento somos más exigentes.
TARGET_RECALL = 0.95

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
# ESPACIO DE BÚSQUEDA
# ============================================================

n_estimators_values = [200, 300]

max_depth_values = [3, 5, 7]

learning_rate_values = [
    0.03,
    0.05,
]

scale_pos_weight_values = [
    10,
    15,
    20,
    25,
    30,
    40,
]

thresholds = [
    0.02,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

configurations = list(
    itertools.product(
        n_estimators_values,
        max_depth_values,
        learning_rate_values,
        scale_pos_weight_values,
    )
)


# ============================================================
# INFORMACIÓN DEL EXPERIMENTO
# ============================================================

print(f"Registros totales: {len(X)}")
print(f"Fallas totales: {int(y.sum())}")

print(f"\nObjetivo de Recall: >= {TARGET_RECALL:.0%}")

print(
    f"Configuraciones XGBoost: "
    f"{len(configurations)}"
)

print(
    f"Umbrales por configuración: "
    f"{len(thresholds)}"
)

print(f"Folds: {N_SPLITS}")

print(
    "Entrenamientos reales: "
    f"{len(configurations) * N_SPLITS}"
)

print(
    "Evaluaciones modelo + threshold + fold: "
    f"{len(configurations) * len(thresholds) * N_SPLITS}"
)


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# CREAR PIPELINE
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
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
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

print(
    "\nIniciando búsqueda de alto Recall "
    "con 5-Fold CV...\n"
)


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

    fold_probabilities = []
    fold_targets = []

    # ========================================================
    # ENTRENAR LOS 5 FOLDS
    # ========================================================

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

        fold_probabilities.append(
            probabilities
        )

        fold_targets.append(
            y_fold_val.to_numpy()
        )

    # ========================================================
    # EVALUAR UMBRALES
    # ========================================================

    for threshold in thresholds:
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        total_tp = 0
        total_fn = 0
        total_fp = 0
        total_tn = 0

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

            tn = int(
                (
                    (predictions == 0)
                    & (y_fold_val == 0)
                ).sum()
            )

            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1_scores.append(f1)

            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_tn += tn

        results.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "scale_pos_weight": scale_pos_weight,
                "threshold": threshold,
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
                "recall_max": float(
                    np.max(fold_recalls)
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
                "tn": total_tn,
            }
        )

    print(
        f"[{config_index}/"
        f"{len(configurations)}] "
        f"trees={n_estimators}, "
        f"depth={max_depth}, "
        f"lr={learning_rate}, "
        f"weight={scale_pos_weight}"
    )


# ============================================================
# CANDIDATOS CON RECALL >= 95 %
# ============================================================

candidates = [
    result
    for result in results
    if result["recall_mean"] >= TARGET_RECALL
]


print("\n" + "=" * 68)
print("RESULTADO DE LA BÚSQUEDA")
print("=" * 68)

print(
    "Combinaciones con Recall promedio >= 95%: "
    f"{len(candidates)}"
)


# ============================================================
# SELECCIÓN
# ============================================================

if candidates:
    # El requisito de Recall ya está satisfecho.
    # Dentro de ese grupo buscamos el mejor equilibrio:
    #
    # 1. Mayor F1
    # 2. Mayor Precision
    # 3. Menor variación del Recall

    best_result = max(
        candidates,
        key=lambda result: (
            result["f1_mean"],
            result["precision_mean"],
            -result["recall_std"],
        ),
    )

else:
    # Si no conseguimos 95 %, mostramos el modelo
    # que más se acercó al objetivo.

    best_result = max(
        results,
        key=lambda result: (
            result["recall_mean"],
            result["f1_mean"],
        ),
    )


# ============================================================
# MEJOR RESULTADO
# ============================================================

print("\n" + "=" * 68)

if candidates:
    print(
        "MEJOR CONFIGURACIÓN CON "
        "RECALL PROMEDIO >= 95%"
    )
else:
    print(
        "NO SE ALCANZÓ 95% - "
        "MEJOR RECALL ENCONTRADO"
    )

print("=" * 68)

print(
    f"n_estimators: "
    f"{best_result['n_estimators']}"
)

print(
    f"max_depth: "
    f"{best_result['max_depth']}"
)

print(
    f"learning_rate: "
    f"{best_result['learning_rate']}"
)

print(
    f"scale_pos_weight: "
    f"{best_result['scale_pos_weight']}"
)

print(
    f"threshold: "
    f"{best_result['threshold']:.2f}"
)


# ============================================================
# MÉTRICAS
# ============================================================

print("\nMétricas 5-Fold CV:")

print(
    "Precision: "
    f"{best_result['precision_mean']:.4f} "
    "± "
    f"{best_result['precision_std']:.4f}"
)

print(
    "Recall: "
    f"{best_result['recall_mean']:.4f} "
    "± "
    f"{best_result['recall_std']:.4f}"
)

print(
    "Recall mínimo: "
    f"{best_result['recall_min']:.4f}"
)

print(
    "Recall máximo: "
    f"{best_result['recall_max']:.4f}"
)

print(
    "F1: "
    f"{best_result['f1_mean']:.4f} "
    "± "
    f"{best_result['f1_std']:.4f}"
)


# ============================================================
# MATRIZ ACUMULADA
# ============================================================

print("\nPredicciones out-of-fold acumuladas:")

print(f"TP: {best_result['tp']}")
print(f"FN: {best_result['fn']}")
print(f"FP: {best_result['fp']}")
print(f"TN: {best_result['tn']}")

total_failures = (
    best_result["tp"]
    + best_result["fn"]
)

print(
    "\nFallas detectadas: "
    f"{best_result['tp']}/"
    f"{total_failures}"
)

print(
    "Fallas NO detectadas: "
    f"{best_result['fn']}/"
    f"{total_failures}"
)


# ============================================================
# TOP 10
# ============================================================

if candidates:
    top_candidates = sorted(
        candidates,
        key=lambda result: (
            result["f1_mean"],
            result["precision_mean"],
            -result["recall_std"],
        ),
        reverse=True,
    )[:10]

    print("\n" + "=" * 68)
    print(
        "TOP 10 CON RECALL PROMEDIO >= 95%"
    )
    print("=" * 68)

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
    run_name="xgboost-high-recall-5fold"
):
    mlflow.log_params(
        {
            "n_splits": N_SPLITS,
            "target_recall": TARGET_RECALL,
            "n_estimators": best_result[
                "n_estimators"
            ],
            "max_depth": best_result[
                "max_depth"
            ],
            "learning_rate": best_result[
                "learning_rate"
            ],
            "scale_pos_weight": best_result[
                "scale_pos_weight"
            ],
            "threshold": best_result[
                "threshold"
            ],
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
            "cv_recall_max": best_result[
                "recall_max"
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
            "cv_tn": best_result["tn"],
        }
    )

    mlflow.set_tags(
        {
            "stage": "high-recall-cv",
            "validation_strategy": (
                "StratifiedKFold-5"
            ),
            "objective": (
                "recall>=0.95 then maximize F1"
            ),
            "test_used": "false",
        }
    )


print("\nExperimento de alto Recall terminado.")
print("No se utilizó TEST para seleccionar el modelo.")