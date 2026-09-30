# Predictive Maintenance MLOps

Plataforma de Machine Learning orientada a la detección temprana de fallas en maquinaria industrial utilizando **XGBoost**, experimentación con **MLflow** y prácticas de **MLOps**.

El proyecto utiliza el dataset **AI4I 2020 Predictive Maintenance Dataset** y prioriza la detección de máquinas con riesgo de falla mediante una estrategia enfocada en **Recall**.

Actualmente se encuentra completada la etapa local de exploración, entrenamiento, validación cruzada, optimización y registro del modelo.

---

## Objetivo

Construir progresivamente una plataforma MLOps capaz de:

- procesar datos de maquinaria industrial;
- entrenar modelos de Machine Learning;
- detectar posibles fallas;
- registrar experimentos y modelos;
- versionar modelos mediante MLflow;
- automatizar pipelines de datos y Machine Learning;
- desplegar modelos en Azure;
- almacenar predicciones;
- visualizar resultados mediante Power BI;
- implementar CI/CD mediante GitHub Actions.

---

## Dataset

Se utiliza el **AI4I 2020 Predictive Maintenance Dataset**, disponible en UCI Machine Learning Repository.

El dataset contiene:

- **10,000 registros**
- **339 fallas**
- **9,661 registros sin falla**

La variable objetivo utilizada es:

```text
Machine failure
```

### Variables utilizadas

El modelo utiliza las siguientes características:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

Las columnas relacionadas directamente con tipos específicos de falla no se utilizan como variables predictoras para evitar **data leakage**:

```text
TWF
HDF
PWF
OSF
RNF
```

También se excluyen identificadores como:

```text
UDI
Product ID
```

---

## Problema de desbalance

El dataset presenta un fuerte desbalance de clases:

```text
Sin falla: 96.61%
Con falla:  3.39%
```

Por esta razón, **Accuracy no se utiliza como única métrica para seleccionar el modelo**.

Las principales métricas consideradas son:

- Recall
- Precision
- F1-score
- PR-AUC

En este proyecto se da especial importancia al **Recall**, debido a que una falla real no detectada puede ser más crítica que generar una alerta preventiva adicional.

---

## Exploración de datos

El análisis exploratorio se encuentra en:

```text
notebooks/01_eda_ai4i.ipynb
```

Durante esta etapa se analizaron:

- estructura del dataset;
- distribución de variables;
- valores faltantes;
- registros duplicados;
- distribución de la variable objetivo;
- desbalance de clases;
- variables disponibles para entrenamiento.

---

## Modelo baseline

Como punto de referencia se entrenó inicialmente un modelo de:

```text
Logistic Regression
```

Resultados obtenidos:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9685 |
| Precision | 0.6667 |
| Recall | 0.1471 |
| F1-score | 0.2410 |
| PR-AUC | 0.4335 |

Aunque la Accuracy era elevada, el modelo únicamente detectaba una pequeña parte de las fallas reales.

Esto confirmó la importancia de utilizar métricas adecuadas para datasets desbalanceados.

El experimento se encuentra en:

```text
notebooks/02_baseline_model.ipynb
```

---

## XGBoost

Posteriormente se experimentó con **XGBoost** para mejorar la capacidad de detección de fallas.

Los experimentos iniciales se encuentran en:

```text
notebooks/03_xgboost_experiments.ipynb
```

Se evaluaron diferentes estrategias relacionadas con:

- profundidad de árboles;
- número de estimadores;
- learning rate;
- `scale_pos_weight`;
- threshold de clasificación;
- balance entre Recall y Precision.

---

## Optimización orientada a Recall

Debido al objetivo del proyecto, se realizaron experimentos específicos para aumentar la cantidad de fallas detectadas.

Los experimentos se encuentran organizados en:

```text
scripts/experiments/
```

Incluyen:

```text
optimize_threshold.py
tune_xgboost_recall.py
evaluate_holdout_experiment.py
tune_xgboost_cv.py
tune_xgboost_high_recall.py
```

---

## Validación cruzada

Para evaluar la estabilidad de las configuraciones se utilizó:

```text
Stratified 5-Fold Cross-Validation
```

Una primera búsqueda orientada a mantener un mejor equilibrio entre Recall, Precision y F1 obtuvo aproximadamente:

| Métrica | Resultado |
|---|---:|
| Precision | 0.3332 ± 0.0136 |
| Recall | 0.9027 ± 0.0197 |
| F1-score | 0.4865 ± 0.0134 |
| Recall mínimo | 0.8676 |

Configuración:

```text
n_estimators     = 200
max_depth        = 7
learning_rate    = 0.03
scale_pos_weight = 15
threshold        = 0.10
```

Este perfil detectó mediante predicciones out-of-fold:

```text
306 / 339 fallas
```

---

## Modelo de alta sensibilidad

Posteriormente se realizó una segunda búsqueda estableciendo como objetivo:

```text
Recall promedio >= 95%
```

Se evaluaron:

```text
72 configuraciones XGBoost
360 entrenamientos
3600 evaluaciones modelo + threshold + fold
```

La configuración seleccionada fue:

```text
n_estimators     = 200
max_depth        = 3
learning_rate    = 0.03
subsample        = 0.8
colsample_bytree = 0.8
scale_pos_weight = 25
threshold        = 0.35
```

### Resultados de 5-Fold Cross-Validation

| Métrica | Resultado |
|---|---:|
| Precision | 0.2456 ± 0.0058 |
| Recall | **0.9558 ± 0.0092** |
| Recall mínimo | 0.9412 |
| Recall máximo | 0.9701 |
| F1-score | 0.3907 ± 0.0075 |

Predicciones out-of-fold acumuladas:

```text
True Positives:  324
False Negatives: 15
False Positives: 996
True Negatives:  8665
```

Por lo tanto:

```text
Fallas detectadas:     324 / 339
Fallas no detectadas:   15 / 339
```

El aumento del Recall implica una reducción de Precision, por lo que este modelo representa un perfil de **alta sensibilidad** orientado a reducir fallas no detectadas.

---

## Comparación de perfiles

Durante la experimentación se identificaron dos perfiles principales:

| Perfil | Precision | Recall | F1 |
|---|---:|---:|---:|
| Equilibrado | 0.3332 | 0.9027 | 0.4865 |
| Alta sensibilidad | 0.2456 | **0.9558** | 0.3907 |

El modelo registrado actualmente como versión de alta sensibilidad prioriza Recall.

La elección operativa entre ambos perfiles dependería del costo real asociado a:

- no detectar una falla;
- generar una falsa alarma;
- realizar una inspección preventiva.

---

## MLflow

**MLflow** se utiliza localmente para:

- registrar experimentos;
- almacenar hiperparámetros;
- registrar métricas;
- mantener metadata del entorno;
- registrar modelos;
- versionar modelos mediante Model Registry.

Modelo registrado:

```text
predictive-maintenance-xgboost
```

Actualmente se dispone de:

```text
Version 1
└── modelo inicial

Version 2
└── modelo XGBoost de alta sensibilidad
```

La Version 2 utiliza:

```text
threshold = 0.35
```

y fue entrenada finalmente utilizando los 10,000 registros después de completar la etapa de selección mediante validación cruzada.

---

## Inferencia

El modelo registrado puede cargarse nuevamente desde MLflow para realizar predicciones.

Script:

```text
scripts/inference/predict_model.py
```

La inferencia obtiene primero la probabilidad:

```python
probability = model.predict_proba(data)[:, 1]
```

y posteriormente aplica explícitamente el threshold seleccionado:

```python
prediction = (probability >= 0.35).astype(int)
```

Esto evita utilizar accidentalmente el threshold estándar de `0.50`.

---

## Estructura del proyecto

```text
mantenimiento-predictivo-mlops/
│
├── .github/
│   └── workflows/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── notebooks/
│   ├── 01_eda_ai4i.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_xgboost_experiments.ipynb
│
├── scripts/
│   ├── experiments/
│   │   ├── evaluate_holdout_experiment.py
│   │   ├── optimize_threshold.py
│   │   ├── tune_xgboost_cv.py
│   │   ├── tune_xgboost_high_recall.py
│   │   └── tune_xgboost_recall.py
│   │
│   ├── inference/
│   │   └── predict_model.py
│   │
│   └── training/
│       ├── register_model.py
│       └── train_mlflow.py
│
├── src/
│   └── maintenance_ml/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── utils/
│
├── tests/
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Tecnologías actuales

### Machine Learning

- Python 3.12
- pandas
- NumPy
- scikit-learn
- XGBoost

### Experimentación y MLOps

- MLflow
- MLflow Model Registry

### Calidad de código

- Ruff
- pytest

### Desarrollo

- Git
- GitHub
- VS Code
- Jupyter

---

## Ejecución local

### 1. Crear entorno virtual

El proyecto utiliza Python 3.12.

Con `uv`:

```powershell
uv venv .venv --python 3.12
```

### 2. Activar el entorno

En PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
uv pip install -e ".[dev]"
```

### 4. Iniciar MLflow

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

La interfaz estará disponible localmente en:

```text
http://127.0.0.1:5000
```

### 5. Ejecutar inferencia

Con MLflow ejecutándose y el modelo registrado localmente:

```powershell
python scripts/inference/predict_model.py
```

---

## Calidad de código

Para comprobar los scripts:

```powershell
ruff check scripts
```

Las pruebas automatizadas se incorporarán progresivamente en:

```text
tests/
```

---

## Arquitectura MLOps objetivo

La evolución prevista del proyecto es:

```text
Datos / sensores simulados
          │
          ▼
Azure Databricks
   PySpark + Delta Lake
          │
          ▼
Feature Engineering / Feature Store
          │
          ▼
Entrenamiento XGBoost
          │
          ▼
MLflow
Experiment Tracking + Model Registry
          │
          ▼
Azure Machine Learning
          │
          ▼
Batch Endpoint
          │
          ▼
SQL Server
          │
          ▼
Power BI
```

La orquestación y automatización se incorporarán mediante:

```text
Azure Data Factory
GitHub Actions
```

---

## Próximas etapas

Las siguientes fases del proyecto son:

1. Integración con Azure Databricks.
2. Procesamiento mediante PySpark.
3. Persistencia de datos mediante Delta Lake.
4. Implementación de Feature Engineering / Feature Store.
5. Integración con Azure Machine Learning.
6. Despliegue mediante Batch Endpoint.
7. Orquestación mediante Azure Data Factory.
8. Implementación de CI/CD con GitHub Actions.
9. Persistencia de predicciones en SQL Server.
10. Dashboard de mantenimiento predictivo en Power BI.
11. Incorporación de pruebas automatizadas adicionales.
12. Documentación de arquitectura y despliegue.

---

## Consideraciones metodológicas

Los resultados reportados para la selección de los modelos corresponden a **Stratified 5-Fold Cross-Validation**.

Durante la fase inicial también se utilizó una separación train/test para experimentación. Posteriormente, las búsquedas mediante validación cruzada utilizaron el dataset completo.

Por esta razón, los resultados de CV se presentan como estimaciones **out-of-fold del proceso de desarrollo**, y el antiguo conjunto test no se presenta como una evaluación independiente posterior a todo el proceso de tuning.

Para una estimación completamente independiente del rendimiento final sería necesario utilizar un nuevo conjunto externo no utilizado durante el desarrollo o aplicar un esquema adicional como nested cross-validation.

---

## Estado actual

```text
[✓] Análisis exploratorio
[✓] Modelo baseline
[✓] XGBoost
[✓] Optimización de hiperparámetros
[✓] Optimización de threshold
[✓] Stratified 5-Fold Cross-Validation
[✓] Optimización para alto Recall
[✓] MLflow Experiment Tracking
[✓] MLflow Model Registry
[✓] Modelo Version 2 registrado
[✓] Inferencia local desde MLflow

[ ] Azure Databricks
[ ] Delta Lake
[ ] Feature Store
[ ] Azure Machine Learning
[ ] Batch Endpoint
[ ] Azure Data Factory
[ ] GitHub Actions CI/CD
[ ] SQL Server
[ ] Power BI
```

---

## Autor

**Miguel Yupanqui**

Proyecto de portafolio orientado a Machine Learning Engineering y MLOps.