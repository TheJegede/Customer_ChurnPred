import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
from quality import check_data_quality


def _make_valid_df(n=200):
    """Minimal valid Telco-schema DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customerID":       [f"ID-{i}" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], n),
        "SeniorCitizen":    rng.integers(0, 2, n).astype("int64"),
        "Partner":          rng.choice(["Yes", "No"], n),
        "Dependents":       rng.choice(["Yes", "No"], n),
        "tenure":           rng.integers(1, 72, n).astype("int64"),
        "PhoneService":     rng.choice(["Yes", "No"], n),
        "MultipleLines":    rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService":  rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":   rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":     rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":      rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":      rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":  rng.choice(["Yes", "No", "No internet service"], n),
        "Contract":         rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod":    rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges":   rng.uniform(20, 100, n).astype("float64"),
        "TotalCharges":     [str(round(v, 2)) for v in rng.uniform(20, 5000, n)],
        "Churn":            rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


class TestQualityGatePasses:
    def test_success_flag(self):
        result = check_data_quality(_make_valid_df())
        assert result["success"] is True

    def test_no_failures(self):
        result = check_data_quality(_make_valid_df())
        assert result["failures"] == []

    def test_statistics_keys_present(self):
        result = check_data_quality(_make_valid_df())
        stats = result["statistics"]
        assert "total_rows" in stats
        assert "total_columns" in stats
        assert "null_rates_by_column" in stats

    def test_row_count_reported(self):
        df = _make_valid_df(n=300)
        result = check_data_quality(df)
        assert result["statistics"]["total_rows"] == 300


class TestQualityGateCatchesBrokenData:
    def test_too_few_rows(self):
        df = _make_valid_df(n=50)   # below MIN_ROWS_CRITICAL=100
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("rows" in f.lower() for f in result["failures"])

    def test_missing_required_column(self):
        df = _make_valid_df().drop(columns=["Churn"])
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("Churn" in f or "Missing" in f for f in result["failures"])

    def test_high_null_rate_triggers_failure(self):
        df = _make_valid_df(n=200)
        # Inject >50% nulls into a required column
        df.loc[:120, "MonthlyCharges"] = np.nan
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("MonthlyCharges" in f for f in result["failures"])

    def test_negative_tenure_triggers_failure(self):
        df = _make_valid_df(n=200)
        df.loc[0, "tenure"] = -5
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("tenure" in f for f in result["failures"])

    def test_single_class_target_triggers_failure(self):
        df = _make_valid_df(n=200)
        df["Churn"] = "No"   # only one class
        result = check_data_quality(df)
        assert result["success"] is False
