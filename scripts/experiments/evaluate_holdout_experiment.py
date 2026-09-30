import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
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
# CONFIGURACIÓN FINAL SELECCIONADA EN VALIDATION
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance"

N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.05
SCALE_POS_WEIGHT = 15

# Umbral seleccionado exclusivamente usando VALIDATION
THRESHOLD = 0.40

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
# TRAIN / TEST
# ============================================================

# Se reproduce exactamente la separación original.
# TEST contiene el 20 % reservado.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Train final: {X_train.shape}")
print(f"Test final: {X_test.shape}")
print(f"Fallas reales en TEST: {int(y_test.sum())}")


# ============================================================
# PIPELINE FINAL
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
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=SCALE_POS_WEIGHT,
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
# ENTRENAMIENTO FINAL
# ============================================================

print("\nEntrenando modelo final con todo TRAIN...")

model.fit(X_train, y_train)


# ============================================================
# EVALUACIÓN ÚNICA EN TEST
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

# NO usamos model.predict(), porque utilizaría umbral 0.50.
y_pred = (y_probability >= THRESHOLD).astype(int)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

pr_auc = average_precision_score(
    y_test,
    y_probability,
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
).ravel()

total_failures = int(y_test.sum())


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADO FINAL EN TEST")
print("=" * 60)

print("\nConfiguración:")

print(f"n_estimators: {N_ESTIMATORS}")
print(f"max_depth: {MAX_DEPTH}")
print(f"learning_rate: {LEARNING_RATE}")
print(f"scale_pos_weight: {SCALE_POS_WEIGHT}")
print(f"threshold: {THRESHOLD:.2f}")

print("\nMétricas:")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print("\nMatriz de confusión:")

print(f"TP: {tp}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"TN: {tn}")

print(
    "\nFallas detectadas: "
    f"{tp}/{total_failures}"
)

print(
    "Fallas NO detectadas: "
    f"{fn}/{total_failures}"
)


# ============================================================
# REGISTRAR RESULTADO FINAL EN MLFLOW
# ============================================================

with mlflow.start_run(run_name="final-recall-model-test"):
    mlflow.log_params(
        {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "threshold": THRESHOLD,
        }
    )

    mlflow.log_metrics(
        {
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_pr_auc": pr_auc,
            "test_tp": int(tp),
            "test_fn": int(fn),
            "test_fp": int(fp),
            "test_tn": int(tn),
        }
    )

    mlflow.set_tags(
        {
            "stage": "final-test",
            "selection_source": "validation",
            "test_usage": "final-evaluation-only",
        }
    )


print("\nEvaluación final terminada.")