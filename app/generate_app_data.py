"""
Generates artefacts consumed by the Streamlit dashboard.
Run once before starting the app (or any time the model changes):

    python app/generate_app_data.py

Outputs
-------
data/predictions.csv   — test-set rows (cleaned features) + model predictions
data/model_results.json — model comparison table with narrative notes
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
CLEANED_CSV      = PROJECT_ROOT / "data" / "cleaned.csv"
FEATURES_CSV     = PROJECT_ROOT / "data" / "features.csv"
PRODUCTION_MODEL = PROJECT_ROOT / "models" / "production_model.pkl"
OUT_PREDICTIONS  = PROJECT_ROOT / "data" / "predictions.csv"
OUT_RESULTS      = PROJECT_ROOT / "data" / "model_results.json"

TARGET       = "Churn"
TEST_SIZE    = 0.20
RANDOM_STATE = 42


def main() -> None:
    df_features = pd.read_csv(FEATURES_CSV)
    df_cleaned  = pd.read_csv(CLEANED_CSV)

    # Reproduce the exact split used during training
    X = df_features.drop(columns=["customerID", TARGET])
    y = LabelEncoder().fit_transform(df_features[TARGET])   # No=0, Yes=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Verify row alignment between the two CSVs before slicing
    assert (df_features["customerID"].values == df_cleaned["customerID"].values).all(), \
        "Row order mismatch between features.csv and cleaned.csv — cannot align."

    test_idx  = X_test.index
    df_test   = df_cleaned.iloc[test_idx].copy().reset_index(drop=True)

    # Run production model
    model  = joblib.load(PRODUCTION_MODEL)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    df_test["predicted_churn"]   = np.where(y_pred == 1, "Yes", "No")
    df_test["churn_probability"] = y_prob.round(4)

    df_test.to_csv(OUT_PREDICTIONS, index=False)
    print(f"Saved predictions.csv  : {len(df_test):,} rows")

    # ------------------------------------------------------------------
    # Model results — sourced from recorded runs
    # ------------------------------------------------------------------
    model_results = {
        "dataset": {
            "total_rows":          7043,
            "features_engineered": 12,
            "features_selected":   29,
            "churn_rate":          0.265,
        },
        "models": [
            {
                "name":           "Logistic Regression",
                "label":          "Baseline",
                "cv_auc":         0.8468,
                "cv_std":         None,
                "test_auc":       0.8478,
                "test_f1":        0.6205,
                "test_recall":    0.7914,
                "test_precision": 0.5103,
                "test_accuracy":  0.7431,
                "train_time":     "< 1s",
                "is_winner":      False,
                "notes": "Strong linear baseline. class_weight=balanced compensates for 73/27 split. Already competitive because the engineered features linearise the churn signal well."
            },
            {
                "name":           "Random Forest",
                "label":          "Candidate",
                "cv_auc":         0.8299,
                "cv_std":         0.0092,
                "test_auc":       0.8207,
                "test_f1":        0.5204,
                "test_recall":    0.4599,
                "test_precision": 0.5993,
                "test_accuracy":  0.7814,
                "train_time":     "0.4s",
                "is_winner":      False,
                "notes": "High precision (0.60) but poor recall (0.46) — misses over half of churners. Unacceptable for a retention team whose cost of missing a churner far exceeds a wasted call."
            },
            {
                "name":           "XGBoost (default)",
                "label":          "Candidate",
                "cv_auc":         0.8347,
                "cv_std":         0.0103,
                "test_auc":       0.8312,
                "test_f1":        0.6129,
                "test_recall":    0.7219,
                "test_precision": 0.5325,
                "test_accuracy":  0.7600,
                "train_time":     "0.4s",
                "is_winner":      False,
                "notes": "Best default tree model. scale_pos_weight=2.77 handles class imbalance. Solid starting point — Optuna tuning pushed it further."
            },
            {
                "name":           "XGBoost (Optuna tuned)",
                "label":          "Winner",
                "cv_auc":         0.8498,
                "cv_std":         0.0118,
                "test_auc":       0.8481,
                "test_f1":        0.6311,
                "test_recall":    0.8075,
                "test_precision": 0.5180,
                "test_accuracy":  0.7495,
                "train_time":     "0.3s",
                "is_winner":      True,
                "notes": "30-trial Optuna TPE search. Best: depth=3, lr=0.024, n=252. Shallow trees prevent overfitting on 5 k training rows. Beats baseline on every metric."
            },
        ],
        "winner": "XGBoost (Optuna tuned)",
        "winner_reasons": [
            "Highest recall (0.808) — catches the most churners, directly reducing revenue loss",
            "Best CV AUC (0.850) — most reliable churn-risk ranking across all decision thresholds",
            "Native feature importances — retention team can see exactly WHY each customer is flagged",
            "Fastest inference of all tree models (0.3 s fit) — trivial to productionise",
        ],
    }

    with open(OUT_RESULTS, "w") as f:
        json.dump(model_results, f, indent=2)
    print("Saved model_results.json")


if __name__ == "__main__":
    main()
