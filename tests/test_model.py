import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "production_model.pkl"


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"production_model.pkl not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def _make_sample_input(n=10):
    """
    Minimal feature row matching the schema that production_model.pkl was trained on.
    Covers every column that the pipeline's ColumnTransformer expects.
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "gender":                       rng.choice(["Male", "Female"], n),
        "SeniorCitizen":                rng.integers(0, 2, n),
        "Partner":                      rng.choice(["Yes", "No"], n),
        "Dependents":                   rng.choice(["Yes", "No"], n),
        "tenure":                       rng.integers(1, 72, n),
        "PhoneService":                 rng.choice(["Yes", "No"], n),
        "MultipleLines":                rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService":              rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":               rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":                 rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":             rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":                  rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":                  rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":              rng.choice(["Yes", "No", "No internet service"], n),
        "Contract":                     rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling":             rng.choice(["Yes", "No"], n),
        "PaymentMethod":                rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges":               rng.uniform(20.0, 100.0, n),
        "TotalCharges":                 rng.uniform(20.0, 5000.0, n),
        "has_social_ties":              rng.integers(0, 2, n),
        "contract_risk_score":          rng.integers(0, 3, n),
        "is_fiber_optic":               rng.integers(0, 2, n),
        "num_services":                 rng.integers(0, 9, n),
        "num_protection_services":      rng.integers(0, 5, n),
        "tenure_group":                 rng.choice(["new", "developing", "established", "loyal"], n),
        "monthly_charges_zscore":       rng.uniform(-3.0, 3.0, n),
        "monthly_charges_quartile":     rng.choice(["Q1", "Q2", "Q3", "Q4"], n),
        "monthly_charges_per_service":  rng.uniform(5.0, 50.0, n),
        "tenure_x_contract_risk":       rng.integers(0, 144, n),
        "charges_to_tenure_ratio":      rng.uniform(0.5, 100.0, n),
        "total_to_monthly_ratio":       rng.uniform(0.0, 70.0, n),
    })


class TestModelLoads:
    def test_model_has_predict(self, model):
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")

    def test_model_is_pipeline(self, model):
        from sklearn.pipeline import Pipeline
        assert isinstance(model, Pipeline)


class TestModelPredictions:
    def test_predict_returns_array(self, model):
        X = _make_sample_input(n=5)
        preds = model.predict(X)
        assert len(preds) == 5

    def test_predict_binary_output(self, model):
        X = _make_sample_input(n=20)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, model):
        X = _make_sample_input(n=10)
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2)

    def test_predict_proba_sums_to_one(self, model):
        X = _make_sample_input(n=15)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, model):
        X = _make_sample_input(n=20)
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_single_row_prediction(self, model):
        X = _make_sample_input(n=1)
        pred = model.predict(X)
        assert pred[0] in (0, 1)

    def test_churn_proba_column(self, model):
        """Column index 1 is churn=Yes; spot-check it's a valid probability."""
        X = _make_sample_input(n=5)
        churn_proba = model.predict_proba(X)[:, 1]
        assert ((churn_proba >= 0) & (churn_proba <= 1)).all()
