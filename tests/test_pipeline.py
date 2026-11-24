"""Unit tests for the Student Performance ML pipeline."""

from src.pipeline import generate_uci_like_dataset, predict_student, train_models


def test_dataset_schema():
    df = generate_uci_like_dataset(n_samples=100)
    assert df.shape == (100, 6)
    assert df.isna().sum().sum() == 0


def test_model_training_and_prediction():
    df = generate_uci_like_dataset(n_samples=300)
    bundle = train_models(df)
    assert bundle["metrics"]["accuracy"] >= 0.75
    assert bundle["metrics"]["r2"] >= 0.65

    out = predict_student(bundle, studytime=3, failures=0, absences=2, g1=14.0, g2=15.0)
    assert 0.0 <= out["predicted_g3"] <= 20.0
    assert 0.0 <= out["pass_probability"] <= 100.0
