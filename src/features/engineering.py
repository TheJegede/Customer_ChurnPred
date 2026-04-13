import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_CSV = PROJECT_ROOT / "data" / "cleaned.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag(series: pd.Series, positive_value: str = "Yes") -> pd.Series:
    """Convert a Yes/No column to 0/1 integer."""
    return (series.astype(str).str.strip() == positive_value).astype(int)


def _coerce_total_charges(df: pd.DataFrame) -> pd.Series:
    """TotalCharges arrives as object with empty strings — coerce to float."""
    return pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Feature categories
# ---------------------------------------------------------------------------

def _domain_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-specific features derived from business understanding of churn.
    Each captures a known churn lever in the telecom industry.
    """

    # Customers with a partner or dependents have higher switching costs —
    # shared accounts and family plans make leaving more disruptive.
    out["has_social_ties"] = (
        (_flag(df["Partner"]) | _flag(df["Dependents"])).astype(int)
    )

    # Month-to-month customers can leave without penalty; longer contracts
    # create exit barriers. Ordinal score ranks risk: 2 = highest churn risk.
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    out["contract_risk_score"] = df["Contract"].map(contract_map).fillna(2).astype(int)

    # Fiber optic customers churn at significantly higher rates in this dataset —
    # they're typically more tech-savvy, price-aware, and have more competitive
    # alternatives than DSL or no-internet customers.
    out["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Count of active add-on services. More services = deeper product lock-in,
    # higher switching costs, and greater perceived value from the provider.
    addon_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    out["num_services"] = sum(_flag(df[col]) for col in addon_cols)

    # Protection services (security, backup, device, support) specifically reduce
    # the pain of switching — customers lose these bundles if they leave.
    protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    out["num_protection_services"] = sum(_flag(df[col]) for col in protection_cols)

    return out


def _statistical_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical transformations that normalise skewed distributions or
    bucket continuous features into interpretable segments.
    """
    total_charges = _coerce_total_charges(df)

    # Tenure groups capture the customer lifecycle stage. New customers (0–12 mo)
    # are at highest risk; loyal customers (49+ mo) have proved their stickiness.
    out["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["new", "developing", "established", "loyal"],
    ).astype(str)

    # Monthly charges are right-skewed. Standardising makes the scale comparable
    # across models that are sensitive to feature magnitude (logistic, SVM, KNN).
    monthly_mean = df["MonthlyCharges"].mean()
    monthly_std = df["MonthlyCharges"].std()
    out["monthly_charges_zscore"] = (
        (df["MonthlyCharges"] - monthly_mean) / monthly_std
    ).round(4)

    # Quartile bucket of MonthlyCharges gives a non-parametric view of price tier
    # and is robust to the right skew — Q4 = premium, Q1 = budget customers.
    out["monthly_charges_quartile"] = pd.qcut(
        df["MonthlyCharges"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    return out


def _interaction_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features where two signals together are more predictive than either alone.
    These capture non-linear relationships that tree models exploit well and
    that linear models otherwise miss.
    """
    total_charges = _coerce_total_charges(df)

    # Cost per service: what a customer pays relative to how many services they use.
    # A high ratio means they're paying a lot for few services — price-value mismatch
    # is a primary churn trigger. +1 avoids division by zero.
    out["monthly_charges_per_service"] = (
        df["MonthlyCharges"] / (out["num_services"] + 1)
    ).round(4)

    # Combines contract risk with tenure. A new customer on month-to-month
    # (score=2, tenure=1) scores 2; a loyal customer on a two-year plan scores 0.
    # This compound signal identifies the single highest-risk cohort.
    out["tenure_x_contract_risk"] = (df["tenure"] * out["contract_risk_score"]).astype(int)

    # Monthly charges divided by tenure approximates "effective price per month of
    # relationship". New customers paying high rates feel less valued than long-tenured
    # customers who may have negotiated discounts — higher ratio → higher churn risk.
    out["charges_to_tenure_ratio"] = (
        df["MonthlyCharges"] / (df["tenure"] + 1)
    ).round(4)

    # Ratio of total spend to monthly charge approximates implied tenure.
    # When much lower than actual tenure, it may signal mid-contract plan downgrades
    # or account credits — a sign of prior dissatisfaction.
    out["total_to_monthly_ratio"] = (
        total_charges / (df["MonthlyCharges"] + 1)
    ).round(4)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """
    Remove redundant and near-constant numeric features.

    Two-pass filter:
      1. Correlation: for each pair with |r| > corr_threshold, drop the
         second feature (keep first encountered in column order).
      2. Variance: drop features whose variance is below
         variance_threshold_pct * mean_variance_across_all_numeric_features.

    Returns (selected_feature_names, reduced_dataframe).
    Non-numeric columns are always kept and excluded from selection logic.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

    dropped: dict[str, str] = {}  # col -> reason

    # ------------------------------------------------------------------
    # Pass 1 — correlation filter
    # ------------------------------------------------------------------
    corr_matrix = df[numeric_cols].corr().abs()
    # Upper triangle mask: we only need each pair once
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

    corr_drops = set()
    for col in upper.columns:
        if col in corr_drops:
            continue
        redundant = upper.index[upper[col] > corr_threshold].tolist()
        for r in redundant:
            if r not in corr_drops:
                partner = col
                r_val = corr_matrix.loc[r, partner]
                dropped[r] = f"correlation with '{partner}' = {r_val:.4f} > {corr_threshold}"
                corr_drops.add(r)

    surviving_numeric = [c for c in numeric_cols if c not in corr_drops]

    # ------------------------------------------------------------------
    # Pass 2 — variance filter
    # ------------------------------------------------------------------
    variances = df[surviving_numeric].var()
    # Use median rather than mean so that one high-scale column (e.g. TotalCharges)
    # does not inflate the threshold and incorrectly eliminate binary/small-range features.
    median_variance = variances.median()
    threshold = variance_threshold_pct * median_variance

    variance_drops = variances[variances < threshold].index.tolist()
    for col in variance_drops:
        dropped[col] = (
            f"variance = {variances[col]:.6f} < threshold "
            f"({variance_threshold_pct:.0%} × median_var {median_variance:.4f} = {threshold:.6f})"
        )

    selected_numeric = [c for c in surviving_numeric if c not in variance_drops]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if dropped:
        print(f"Dropped {len(dropped)} feature(s):")
        for col, reason in dropped.items():
            print(f"  - {col}: {reason}")
    else:
        print("No features dropped — all passed correlation and variance filters.")

    selected = non_numeric_cols + selected_numeric
    print(f"\nSelected {len(selected_numeric)} / {len(numeric_cols)} numeric features "
          f"(+ {len(non_numeric_cols)} non-numeric kept as-is) "
          f"[variance threshold = {threshold:.6f}]")

    return selected, df[selected]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 12 new features across domain, statistical, and interaction categories.

    Returns a new DataFrame containing all original columns plus engineered features.
    The input DataFrame is not modified.
    """
    out = df.copy()

    out = _domain_features(df, out)
    out = _statistical_features(df, out)
    out = _interaction_features(df, out)

    engineered = [c for c in out.columns if c not in df.columns]
    print(f"Engineered {len(engineered)} new features:")
    for name in engineered:
        print(f"  + {name}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else CLEANED_CSV
    df = pd.read_csv(path)
    print(f"Loaded cleaned data: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    df_feat = create_features(df)

    print(f"\nBefore: {df.shape[1]} columns")
    print(f"After engineering: {df_feat.shape[1]} columns")

    print(f"\n--- Feature Selection ---")
    selected_cols, df_selected = select_features(df_feat)
    print(f"After selection:   {df_selected.shape[1]} columns")
