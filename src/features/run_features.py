import time
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_CSV = PROJECT_ROOT / "data" / "cleaned.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"


def main() -> None:
    t0 = time.time()

    # Load
    print(f"Loading {CLEANED_CSV.name} ...")
    df = pd.read_csv(CLEANED_CSV)
    print(f"  Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Engineer features
    print("\nEngineering features ...")
    t1 = time.time()
    from engineering import create_features, select_features
    df_feat = create_features(df)
    print(f"  Done in {time.time() - t1:.2f}s  ->  {df_feat.shape[1]} columns")

    # Select features
    print("\nSelecting features ...")
    t2 = time.time()
    selected_cols, df_selected = select_features(df_feat)
    print(f"  Done in {time.time() - t2:.2f}s  ->  {df_selected.shape[1]} columns")

    # Save
    print(f"\nSaving to {FEATURES_CSV.name} ...")
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(FEATURES_CSV, index=False)
    print(f"  Saved {df_selected.shape[0]:,} rows x {df_selected.shape[1]} columns")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Before : {df.shape[0]:,} rows x {df.shape[1]:>2} columns")
    print(f"  After  : {df_selected.shape[0]:,} rows x {df_selected.shape[1]:>2} columns")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print(f"{'='*50}")

    print(f"\nKept features ({len(selected_cols)}):")
    for col in selected_cols:
        dtype = str(df_selected[col].dtype)
        print(f"  {col:<35} {dtype}")


if __name__ == "__main__":
    main()
