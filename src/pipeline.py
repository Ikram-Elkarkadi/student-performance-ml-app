"""Reproducible Scikit-learn pipeline for Student Performance prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]


def generate_uci_like_dataset(n_samples: int = 649, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset matching the UCI Student Performance schema."""
    rng = np.random.default_rng(random_state)
    studytime = rng.integers(1, 5, size=n_samples)
    failures = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.78, 0.14, 0.05, 0.03])
    absences = rng.integers(0, 25, size=n_samples)
    g1 = np.clip(rng.normal(12.2, 3.1, size=n_samples), 0, 20).round(1)
    g2 = np.clip(g1 + rng.normal(0.3, 1.4, size=n_samples), 0, 20).round(1)

    noise = rng.normal(0, 0.9, size=n_samples)
    g3 = np.clip(0.25 * g1 + 0.72 * g2 + 0.35 * studytime - 0.6 * failures - 0.08 * absences + noise, 0, 20).round(1)

    return pd.DataFrame({
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2,
        "G3": g3,
    })


def train_models(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Train Random Forest Regressor & Classifier and return models with evaluation metrics."""
    X = df[FEATURE_COLUMNS]
    y_reg = df["G3"]
    y_clf = (df["G3"] >= 10.0).astype(int)

    X_train, X_test, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=random_state
    )

    reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state)),
    ])
    clf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)),
    ])

    reg_pipe.fit(X_train, y_reg_tr)
    clf_pipe.fit(X_train, y_clf_tr)

    reg_preds = reg_pipe.predict(X_test)
    clf_preds = clf_pipe.predict(X_test)

    return {
        "regressor": reg_pipe,
        "classifier": clf_pipe,
        "metrics": {
            "r2": round(float(r2_score(y_reg_te, reg_preds)), 4),
            "mae": round(float(mean_absolute_error(y_reg_te, reg_preds)), 4),
            "accuracy": round(float(accuracy_score(y_clf_te, clf_preds)), 4),
            "f1": round(float(f1_score(y_clf_te, clf_preds)), 4),
        },
        "feature_importances": dict(
            zip(FEATURE_COLUMNS, reg_pipe.named_steps["model"].feature_importances_.round(4).tolist())
        ),
    }


def predict_student(bundle: dict, studytime: int, failures: int, absences: int, g1: float, g2: float) -> dict:
    """Predict final score (0-20) and pass/fail probability for a single student."""
    row = pd.DataFrame([{
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2,
    }])
    score = float(np.clip(bundle["regressor"].predict(row)[0], 0.0, 20.0))
    pass_prob = float(bundle["classifier"].predict_proba(row)[0][1])
    return {"predicted_g3": round(score, 2), "pass_probability": round(pass_prob * 100, 1)}


if __name__ == "__main__":
    data = generate_uci_like_dataset()
    result = train_models(data)
    print("Evaluation Metrics:", result["metrics"])
    print("Feature Importances:", result["feature_importances"])
