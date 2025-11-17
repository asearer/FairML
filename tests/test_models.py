# tests/test_models.py

from src.models.example_model import load_model, train_example_model
import pandas as pd


def test_train_example_model():
    model = train_example_model()
    assert model is not None


def test_load_model_after_train(tmp_path, monkeypatch):
    # Redirect model path
    model_path = tmp_path / "model.pkl"
    monkeypatch.setattr(
        "src.models.example_model.MODEL_PATH",
        str(model_path),
    )

    # Train and save
    trained = train_example_model()
    assert model_path.exists()

    # Load
    loaded = load_model()
    assert loaded is not None


def test_model_predicts():
    model = train_example_model()
    df = pd.DataFrame({"x": [1, 2, 3]})
    preds = model.predict(df)
    assert len(preds) == len(df)
