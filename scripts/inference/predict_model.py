import os

import mlflow
import pandas as pd

# ============================================================
# CONFIGURACIÓN DE MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "predictive-maintenance-xgboost"
MODEL_VERSION = "2"

# Umbral seleccionado mediante Stratified 5-Fold CV
DECISION_THRESHOLD = 0.35

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# CARGA DEL MODELO REGISTRADO
# ============================================================

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(
    f"Cargando modelo: {MODEL_NAME}, "
    f"versión {MODEL_VERSION}..."
)

model = mlflow.sklearn.load_model(model_uri)

print("Modelo cargado correctamente.")
print(f"Threshold operativo: {DECISION_THRESHOLD}")


# ============================================================
# DATOS DE EJEMPLO
# ============================================================

sample = pd.DataFrame(
    [
        {
            "Type": "M",
            "Air temperature": 298.1,
            "Process temperature": 308.6,
            "Rotational speed": 1551,
            "Torque": 42.8,
            "Tool wear": 0,
        }
    ]
)


# ============================================================
# PREDICCIÓN
# ============================================================

# Obtenemos la probabilidad de falla.
probability = model.predict_proba(sample)[:, 1]

# Aplicamos explícitamente el threshold seleccionado
# mediante validación cruzada.
prediction = (
    probability >= DECISION_THRESHOLD
).astype(int)


# ============================================================
# RESULTADO
# ============================================================

print("\nDatos enviados al modelo:")
print(sample)

print("\nResultado:")

print(
    f"Probabilidad de falla: "
    f"{probability[0]:.4f}"
)

print(
    f"Threshold utilizado: "
    f"{DECISION_THRESHOLD:.2f}"
)

print(
    f"Predicción final: "
    f"{int(prediction[0])}"
)

if prediction[0] == 1:
    print(
        "Interpretación: posible falla de la máquina."
    )
else:
    print(
        "Interpretación: no se predice una falla "
        "de la máquina."
    )