import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_CSV = PROJECT_ROOT / "data" / "cleaned.csv"

NULL_DROP_THRESHOLD = 0.50  # drop columns with more than 50% nulls

# Columns that should be numeric after cleaning
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Columns that should be string/categorical
CATEGORICAL_COLUMNS = [
    "customerID", "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]

TARGET_COLUMN = "Churn"
IS_TIME_SERIES = False


def _drop_high_null_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > NULL_DROP_THRESHOLD].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def _handle_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Drop rows where target is null
    before = len(df)
    if TARGET_COLUMN in df.columns:
        df = df[df[TARGET_COLUMN].notna()].copy()

    if IS_TIME_SERIES:
        df = df.ffill()
    else:
        df = df.dropna()

    rows_removed = before - len(df)
    return df, rows_removed


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    dupes_removed = before - len(df)
    return df, dupes_removed


def _convert_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    conversion_notes = []

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        original_dtype = str(df[col].dtype)
        converted = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
        n_coerced = converted.isnull().sum() - df[col].isnull().sum()
        df[col] = converted
        # Use int where no fractional part, float otherwise
        non_null = df[col].dropna()
        if (non_null == non_null.astype("int64", errors="ignore")).all():
            df[col] = df[col].astype("Int64")  # nullable integer
        new_dtype = str(df[col].dtype)
        if original_dtype != new_dtype:
            note = f"'{col}': {original_dtype} -> {new_dtype}"
            if n_coerced > 0:
                note += f" ({n_coerced} value(s) coerced to NaN)"
            conversion_notes.append(note)

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        original_dtype = str(df[col].dtype)
        df[col] = df[col].astype(str).str.strip()
        new_dtype = str(df[col].dtype)
        if original_dtype != new_dtype:
            conversion_notes.append(f"'{col}': {original_dtype} -> {new_dtype}")

    return df, conversion_notes


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from quality import check_data_quality

    print(f"Starting shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 1. Drop columns with too many nulls
    df, dropped_cols = _drop_high_null_columns(df)
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} high-null column(s): {dropped_cols}")

    # 2. Handle nulls
    df, null_rows_removed = _handle_nulls(df)
    if null_rows_removed:
        print(f"  Removed {null_rows_removed:,} row(s) with nulls")

    # 3. Drop duplicates
    df, dupes_removed = _drop_duplicates(df)
    if dupes_removed:
        print(f"  Removed {dupes_removed:,} duplicate row(s)")

    # 4. Convert dtypes
    df, conversion_notes = _convert_dtypes(df)
    if conversion_notes:
        print("  Dtype conversions:")
        for note in conversion_notes:
            print(f"    {note}")

    print(f"Cleaned shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 5. Save to data/cleaned.csv
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved: {CLEANED_CSV}")

    # 6. Re-run quality gate
    quality_result = check_data_quality(df)

    return df, quality_result


if __name__ == "__main__":
    import sys
    from loader import load_csv
    from quality import print_report

    path = sys.argv[1] if len(sys.argv) > 1 else None
    raw_df = load_csv(path)

    print(f"\nBefore: {len(raw_df):,} rows x {raw_df.shape[1]} columns")
    print("-" * 50)

    cleaned_df, quality_result = clean_data(raw_df)

    print(f"\nAfter:  {len(cleaned_df):,} rows x {cleaned_df.shape[1]} columns")
    print(f"Rows removed: {len(raw_df) - len(cleaned_df):,}")

    print_report(quality_result)

    if not quality_result["success"]:
        sys.exit(1)
