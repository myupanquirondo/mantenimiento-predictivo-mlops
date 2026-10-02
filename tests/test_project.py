from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_has_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_project_has_source_directory():
    assert (PROJECT_ROOT / "src" / "maintenance_ml").exists()


def test_project_has_training_scripts():
    training_dir = PROJECT_ROOT / "scripts" / "training"

    assert training_dir.exists()
    assert (training_dir / "train_mlflow.py").exists()
    assert (training_dir / "register_model.py").exists()


def test_project_has_inference_script():
    assert (
        PROJECT_ROOT / "scripts" / "inference" / "predict_model.py"
    ).exists()