import pandas as pd
import numpy as np
from pathlib import Path

# Schema definition for the Telco churn dataset
REQUIRED_COLUMNS = {
    "customerID": "object",
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "object",
    "Churn": "object",
}

# Bounds: (min, max) — None means unchecked on that side
NUMERIC_BOUNDS = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, None),
    "MonthlyCharges": (0, 500),
    "TotalCharges": None,  # stored as object; skipped in range check
}

TARGET_COLUMN = "Churn"
MIN_CLASS_SHARE = 0.05   # 5% — below this triggers imbalance warning
IMBALANCE_WARN = 0.20    # 20% — secondary imbalance warning threshold

NULL_CRITICAL = 0.50
NULL_WARN = 0.20
MIN_ROWS_CRITICAL = 100
MIN_ROWS_WARN = 1000


def _check_schema(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")

    for col, expected_dtype in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != expected_dtype:
            warnings.append(
                f"Column '{col}': expected dtype '{expected_dtype}', got '{actual}'"
            )

    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    n = len(df)
    if n < MIN_ROWS_CRITICAL:
        failures.append(f"Only {n} rows — minimum required is {MIN_ROWS_CRITICAL}")
    elif n < MIN_ROWS_WARN:
        warnings.append(f"Only {n} rows — recommend at least {MIN_ROWS_WARN} for reliable modelling")
    return failures, warnings


def _check_null_rates(df: pd.DataFrame) -> tuple[list, list, dict]:
    failures, warnings = [], []
    null_rates = {}

    for col in df.columns:
        rate = df[col].isnull().mean()
        null_rates[col] = round(rate, 4)
        if rate > NULL_CRITICAL:
            failures.append(
                f"Column '{col}' has {rate:.1%} nulls — exceeds critical threshold of {NULL_CRITICAL:.0%}"
            )
        elif rate > NULL_WARN:
            warnings.append(
                f"Column '{col}' has {rate:.1%} nulls — exceeds warning threshold of {NULL_WARN:.0%}"
            )

    return failures, warnings, null_rates


def _check_value_ranges(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []

    for col, bounds in NUMERIC_BOUNDS.items():
        if col not in df.columns or bounds is None:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            warnings.append(f"Column '{col}' has no numeric values to range-check")
            continue

        lo, hi = bounds
        if lo is not None:
            n_below = (series < lo).sum()
            if n_below > 0:
                failures.append(
                    f"Column '{col}': {n_below} value(s) below minimum {lo} "
                    f"(min found: {series.min()})"
                )
        if hi is not None:
            n_above = (series > hi).sum()
            if n_above > 0:
                warnings.append(
                    f"Column '{col}': {n_above} value(s) above expected maximum {hi} "
                    f"(max found: {series.max()})"
                )

    return failures, warnings


def _check_target_distribution(df: pd.DataFrame) -> tuple[list, list, dict]:
    failures, warnings = [], []
    dist = {}

    if TARGET_COLUMN not in df.columns:
        failures.append(f"Target column '{TARGET_COLUMN}' not found — skipping distribution check")
        return failures, warnings, dist

    value_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    total = len(df)
    dist = {str(k): int(v) for k, v in value_counts.items()}
    shares = {str(k): round(v / total, 4) for k, v in value_counts.items()}

    n_classes = len(value_counts)
    if n_classes < 2:
        failures.append(
            f"Target '{TARGET_COLUMN}' has only {n_classes} class — need at least 2"
        )

    minority_share = min(shares.values())
    minority_class = min(shares, key=shares.get)

    if minority_share < MIN_CLASS_SHARE:
        failures.append(
            f"Target '{TARGET_COLUMN}': class '{minority_class}' has only "
            f"{minority_share:.1%} of data — below critical threshold of {MIN_CLASS_SHARE:.0%}"
        )
    elif minority_share < IMBALANCE_WARN:
        warnings.append(
            f"Target '{TARGET_COLUMN}': class '{minority_class}' has only "
            f"{minority_share:.1%} of data — consider class weighting or resampling"
        )

    return failures, warnings, {"class_counts": dist, "class_shares": shares}


def check_data_quality(df: pd.DataFrame) -> dict:
    all_failures = []
    all_warnings = []
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    # 1. Schema
    f, w = _check_schema(df)
    all_failures.extend(f)
    all_warnings.extend(w)

    # 2. Row count
    f, w = _check_row_count(df)
    all_failures.extend(f)
    all_warnings.extend(w)

    # 3. Null rates
    f, w, null_rates = _check_null_rates(df)
    all_failures.extend(f)
    all_warnings.extend(w)
    statistics["total_nulls_by_column"] = {
        col: int(df[col].isnull().sum()) for col in df.columns
    }
    statistics["null_rates_by_column"] = null_rates

    # 4. Value ranges
    f, w = _check_value_ranges(df)
    all_failures.extend(f)
    all_warnings.extend(w)

    # 5. Target distribution
    f, w, target_stats = _check_target_distribution(df)
    all_failures.extend(f)
    all_warnings.extend(w)
    statistics["target_distribution"] = target_stats

    return {
        "success": len(all_failures) == 0,
        "failures": all_failures,
        "warnings": all_warnings,
        "statistics": statistics,
    }


def print_report(result: dict) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"\n{'='*50}")
    print(f"  Data Quality Gate: {status}")
    print(f"{'='*50}")

    stats = result["statistics"]
    print(f"\nRows: {stats['total_rows']:,}   Columns: {stats['total_columns']}")

    if result["failures"]:
        print(f"\nCRITICAL FAILURES ({len(result['failures'])}):")
        for f in result["failures"]:
            print(f"  [FAIL] {f}")

    if result["warnings"]:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"  [WARN] {w}")

    if not result["failures"] and not result["warnings"]:
        print("\n  All checks passed with no warnings.")

    dist = stats.get("target_distribution", {})
    if dist:
        print("\nTarget Distribution:")
        for cls, share in dist.get("class_shares", {}).items():
            count = dist["class_counts"][cls]
            print(f"  {cls}: {count:,} ({share:.1%})")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    import sys
    from loader import load_csv

    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_csv(path)
    result = check_data_quality(df)
    print_report(result)

    if not result["success"]:
        sys.exit(1)
