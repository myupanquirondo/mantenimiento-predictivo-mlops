import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def create_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Type"],
            )
        ],
        remainder="passthrough",
    )

    classifier = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=25,
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


def create_training_data():
    X = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature": 298.1,
                "Process temperature": 308.6,
                "Rotational speed": 1551,
                "Torque": 42.8,
                "Tool wear": 0,
            },
            {
                "Type": "L",
                "Air temperature": 300.0,
                "Process temperature": 310.0,
                "Rotational speed": 1500,
                "Torque": 40.0,
                "Tool wear": 10,
            },
            {
                "Type": "H",
                "Air temperature": 305.0,
                "Process temperature": 315.0,
                "Rotational speed": 1400,
                "Torque": 60.0,
                "Tool wear": 150,
            },
            {
                "Type": "M",
                "Air temperature": 310.0,
                "Process temperature": 320.0,
                "Rotational speed": 1300,
                "Torque": 65.0,
                "Tool wear": 200,
            },
        ]
    )

    y = pd.Series([0, 0, 1, 1])

    return X, y


def test_model_can_train():
    model = create_model()
    X, y = create_training_data()

    model.fit(X, y)

    assert model is not None


def test_model_accepts_expected_features():
    model = create_model()
    X, y = create_training_data()

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (4, 2)


def test_probability_is_valid():
    model = create_model()
    X, y = create_training_data()

    model.fit(X, y)

    probability = model.predict_proba(X)[0, 1]

    assert 0.0 <= probability <= 1.0


def test_prediction_is_binary():
    model = create_model()
    X, y = create_training_data()

    model.fit(X, y)

    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})