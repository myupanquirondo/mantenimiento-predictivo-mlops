# Predictive Maintenance MLOps

Plataforma de Machine Learning para la detección temprana de fallas en maquinaria industrial utilizando **XGBoost**, **MLflow** y prácticas de **MLOps**.

El proyecto utiliza el **AI4I 2020 Predictive Maintenance Dataset** y prioriza la detección de fallas mediante la optimización de **Recall**.

Actualmente se completó la etapa local de exploración, entrenamiento, validación, optimización y registro del modelo.

---

# Objetivo

Construir progresivamente una plataforma MLOps capaz de:

- Procesar datos de maquinaria industrial.
- Entrenar modelos de Machine Learning.
- Detectar posibles fallas.
- Registrar experimentos y métricas.
- Versionar modelos mediante MLflow.
- Automatizar pipelines.
- Desplegar modelos en Azure.
- Almacenar predicciones.
- Visualizar resultados mediante Power BI.
- Implementar CI/CD con GitHub Actions.

---

# Dataset

## AI4I 2020 Predictive Maintenance Dataset

Características:

- 10,000 registros.
- 339 fallas.
- 9,661 registros sin falla.

Variable objetivo:

```text
Machine failure
```

Variables utilizadas:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

Variables excluidas para evitar data leakage:

```text
TWF
HDF
PWF
OSF
RNF
```

También se excluyeron identificadores:

```text
UDI
Product ID
```

---

# Problema de desbalance

El dataset presenta un fuerte desbalance:

```
Sin falla: 96.61%
Con falla: 3.39%
```

Por esta razón, la selección del modelo no se basó únicamente en Accuracy.

Métricas principales:

- Recall
- Precision
- F1-score
- PR-AUC

El objetivo principal fue reducir la cantidad de fallas reales no detectadas.

---

# Modelamiento

## Baseline

Inicialmente se entrenó:

```
Logistic Regression
```

El modelo baseline presentó un Recall reducido, por lo que se evaluó una solución basada en árboles.

---

# XGBoost

Se realizaron experimentos con:

- Número de árboles.
- Profundidad.
- Learning rate.
- `scale_pos_weight`.
- Threshold de clasificación.

Notebook principal:

```
notebooks/03_xgboost_experiments.ipynb
```

---

# Optimización de Recall

Se realizaron búsquedas mediante:

- Ajuste de hiperparámetros.
- Optimización del threshold.
- Stratified 5-Fold Cross Validation.

Scripts utilizados:

```
scripts/experiments/
```

Incluye:

```
optimize_threshold.py
tune_xgboost_recall.py
evaluate_holdout_experiment.py
tune_xgboost_cv.py
tune_xgboost_high_recall.py
```

---

# Modelo final seleccionado

Perfil:

```
Alta sensibilidad
```

Configuración:

```
n_estimators     = 200
max_depth        = 3
learning_rate    = 0.03
subsample        = 0.8
colsample_bytree = 0.8
scale_pos_weight = 25
threshold        = 0.35
```

Resultados obtenidos mediante Cross Validation:

| Métrica | Resultado |
|---|---|
| Precision | 0.2456 |
| Recall | 0.9558 |
| F1-score | 0.3907 |

Detección:

```
324 / 339 fallas detectadas
```

El modelo prioriza detectar la mayor cantidad posible de fallas, aceptando un mayor número de falsas alarmas.

---

# MLflow

MLflow se utiliza para:

- Registrar experimentos.
- Guardar hiperparámetros.
- Registrar métricas.
- Versionar modelos.
- Gestionar Model Registry.

Modelo registrado:

```
predictive-maintenance-xgboost
```

Versiones:

```
Version 1
Modelo inicial

Version 2
Modelo XGBoost high recall
```

---

# Estructura del proyecto

```
mantenimiento-predictivo-mlops/

├── notebooks/
│   ├── 01_eda_ai4i.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_xgboost_experiments.ipynb
│
├── scripts/
│   │
│   ├── experiments/
│   │   ├── evaluate_holdout_experiment.py
│   │   ├── optimize_threshold.py
│   │   ├── tune_xgboost_cv.py
│   │   ├── tune_xgboost_high_recall.py
│   │   └── tune_xgboost_recall.py
│   │
│   ├── training/
│   │   ├── train_mlflow.py
│   │   └── register_model.py
│   │
│   └── inference/
│       └── predict_model.py
│
├── src/
│   └── maintenance_ml/
│
├── tests/
│
├── pyproject.toml
├── .gitignore
└── README.md
```

---

# Tecnologías

## Machine Learning

- Python 3.12
- pandas
- NumPy
- scikit-learn
- XGBoost

## MLOps

- MLflow
- MLflow Model Registry

## Calidad de código

- Ruff
- pytest

## Desarrollo

- Git
- GitHub
- VS Code
- Jupyter

---

# Ejecución local

Crear entorno virtual:

```bash
uv venv .venv --python 3.12
```

Activar entorno:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
uv pip install -e ".[dev]"
```

Iniciar MLflow:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Ejecutar inferencia:

```bash
python scripts/inference/predict_model.py
```

---

# Calidad de código

Ejecutar Ruff:

```bash
ruff check scripts
```

---

# Arquitectura MLOps futura

```
Datos
 |
 v
Azure Databricks
 |
 v
Delta Lake
 |
 v
Feature Store
 |
 v
XGBoost
 |
 v
MLflow
 |
 v
Azure Machine Learning
 |
 v
Batch Endpoint
 |
 v
SQL Server
 |
 v
Power BI
```

Automatización prevista:

```
Azure Data Factory
GitHub Actions
```

---

# Próximas etapas

- Azure Databricks.
- Delta Lake.
- Feature Store.
- Azure Machine Learning.
- Batch Endpoint.
- Azure Data Factory.
- GitHub Actions CI/CD.
- SQL Server.
- Power BI.
- Pruebas automatizadas.

---

# Estado actual

✅ Análisis exploratorio  
✅ Modelo baseline  
✅ XGBoost  
✅ Optimización de hiperparámetros  
✅ Optimización de threshold  
✅ Stratified 5-Fold Cross Validation  
✅ Modelo High Recall  
✅ MLflow Tracking  
✅ MLflow Model Registry  
✅ Modelo Version 2 registrado  
✅ Inferencia local  


Pendiente:

⬜ Azure Databricks  
⬜ Delta Lake  
⬜ Feature Store  
⬜ Azure ML  
⬜ Batch Endpoint  
⬜ CI/CD  
⬜ SQL Server  
⬜ Power BI  


---

# Autor

**Miguel Yupanqui**

Proyecto orientado a Machine Learning Engineering y MLOps.
