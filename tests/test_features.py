import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
from engineering import create_features, select_features

ENGINEERED_FEATURE_COUNT = 12

EXPECTED_FEATURES = [
    "has_social_ties",
    "contract_risk_score",
    "is_fiber_optic",
    "num_services",
    "num_protection_services",
    "tenure_group",
    "monthly_charges_zscore",
    "monthly_charges_quartile",
    "monthly_charges_per_service",
    "tenure_x_contract_risk",
    "charges_to_tenure_ratio",
    "total_to_monthly_ratio",
]


def _make_cleaned_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "customerID":       [f"ID-{i}" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], n),
        "SeniorCitizen":    rng.integers(0, 2, n),
        "Partner":          rng.choice(["Yes", "No"], n),
        "Dependents":       rng.choice(["Yes", "No"], n),
        "tenure":           rng.integers(1, 72, n),
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
        "MonthlyCharges":   rng.uniform(20.0, 100.0, n),
        "TotalCharges":     [str(round(v, 2)) for v in rng.uniform(20, 5000, n)],
        "Churn":            rng.choice(["Yes", "No"], n),
    })


@pytest.fixture(scope="module")
def featured_df():
    return create_features(_make_cleaned_df())


class TestFeatureCount:
    def test_adds_exactly_12_features(self, featured_df):
        base_cols = len(_make_cleaned_df().columns)
        assert featured_df.shape[1] == base_cols + ENGINEERED_FEATURE_COUNT

    def test_all_expected_features_present(self, featured_df):
        for feat in EXPECTED_FEATURES:
            assert feat in featured_df.columns, f"Missing engineered feature: {feat}"

    def test_input_not_mutated(self):
        df = _make_cleaned_df()
        original_cols = list(df.columns)
        create_features(df)
        assert list(df.columns) == original_cols


class TestNoNaNs:
    def test_numeric_features_no_nan(self, featured_df):
        numeric_feats = [
            "has_social_ties", "contract_risk_score", "is_fiber_optic",
            "num_services", "num_protection_services",
            "monthly_charges_zscore", "monthly_charges_per_service",
            "tenure_x_contract_risk", "charges_to_tenure_ratio", "total_to_monthly_ratio",
        ]
        for col in numeric_feats:
            assert featured_df[col].isna().sum() == 0, f"{col} has NaN values"

    def test_categorical_features_no_nan(self, featured_df):
        for col in ["tenure_group", "monthly_charges_quartile"]:
            assert featured_df[col].isna().sum() == 0, f"{col} has NaN values"


class TestFeatureRanges:
    def test_has_social_ties_binary(self, featured_df):
        assert set(featured_df["has_social_ties"].unique()).issubset({0, 1})

    def test_is_fiber_optic_binary(self, featured_df):
        assert set(featured_df["is_fiber_optic"].unique()).issubset({0, 1})

    def test_contract_risk_score_range(self, featured_df):
        assert featured_df["contract_risk_score"].between(0, 2).all()

    def test_num_services_non_negative(self, featured_df):
        assert (featured_df["num_services"] >= 0).all()

    def test_num_services_max(self, featured_df):
        # 8 add-on columns in _domain_features — can't exceed that
        assert (featured_df["num_services"] <= 8).all()

    def test_num_protection_services_range(self, featured_df):
        assert featured_df["num_protection_services"].between(0, 4).all()

    def test_monthly_charges_per_service_positive(self, featured_df):
        assert (featured_df["monthly_charges_per_service"] > 0).all()

    def test_tenure_group_valid_labels(self, featured_df):
        valid = {"new", "developing", "established", "loyal"}
        actual = set(featured_df["tenure_group"].unique())
        assert actual.issubset(valid)

    def test_monthly_charges_quartile_valid_labels(self, featured_df):
        valid = {"Q1", "Q2", "Q3", "Q4"}
        actual = set(featured_df["monthly_charges_quartile"].unique())
        assert actual.issubset(valid)

    def test_tenure_x_contract_risk_non_negative(self, featured_df):
        assert (featured_df["tenure_x_contract_risk"] >= 0).all()


class TestSelectFeatures:
    def test_returns_dataframe(self, featured_df):
        _, reduced = select_features(featured_df)
        assert isinstance(reduced, pd.DataFrame)

    def test_no_columns_added(self, featured_df):
        _, reduced = select_features(featured_df)
        assert reduced.shape[1] <= featured_df.shape[1]
