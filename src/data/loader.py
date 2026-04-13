import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_CSV = PROJECT_ROOT / "Customer Churn Prediction" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_csv(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        resolved = DEFAULT_CSV
    else:
        p = Path(path)
        resolved = p if p.is_absolute() else DATA_DIR / p
    df = pd.read_csv(resolved)
    print(f"Loaded: {resolved}")
    return df


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows} rows x {cols} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    print("\nColumn Names and Data Types:")
    print(df.dtypes.to_string())


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric_df.agg(["mean", "std", "min", "max"])
    print("\nSummary Statistics (numeric columns):")
    print(stats.to_string())


def print_missing_values(df: pd.DataFrame) -> None:
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": missing_pct,
    })
    missing = missing[missing["missing_count"] > 0]
    if missing.empty:
        print("\nMissing Values: none")
    else:
        print("\nMissing Values:")
        print(missing.to_string())


def profile(df: pd.DataFrame) -> None:
    print_shape(df)
    print_dtypes(df)
    print_summary_stats(df)
    print_missing_values(df)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_csv(path)
    profile(df)
