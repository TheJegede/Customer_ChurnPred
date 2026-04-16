"""
Trains and compares three candidate models against the logistic regression baseline.

Model selection rationale (retention team use case):
  - Random Forest    : ensemble of decision trees; native feature importance that maps
                       directly to business levers (which services, contract type, tenure
                       segment drove the prediction); robust to outliers and skew.
  - XGBoost          : gradient-boosted trees; typically best AUC on structured tabular
                       churn data; handles class imbalance via scale_pos_weight; SHAP
                       values available for per-customer explanations.
  - LightGBM         : histogram-based gradient boosting; faster than XGBoost on this
                       dataset size with similar accuracy; leaf-wise growth captures
                       complex interactions (e.g. tenure x contract risk) efficiently.

Primary CV metric: AUC-ROC — measures ranking quality across all decision thresholds,
letting the retention team tune the operating point to match their capacity.
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# LightGBM emits a spurious sklearn feature-name warning during cross_val_score
# because ColumnTransformer outputs a numpy array (no column names). Suppressed here
# as it's a known sklearn/LightGBM integration issue and does not affect results.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

DROP_COLS = ["customerID"]
TARGET = "Churn"
CV_FOLDS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """
    Shared preprocessor for all three models.
    Trees don't require scaling, so numeric columns only get imputation.
    OHE is applied to categoricals (trees handle this fine; avoids ordinal assumptions).
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            (
                "num",
                # TotalCharges has 11 NaN rows (tenure=0 new customers) — fill 0.0
                SimpleImputer(strategy="constant", fill_value=0.0),
                numeric_cols,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(scale_pos_weight: float) -> dict[str, object]:
    """
    Returns classifiers keyed by display name.
    scale_pos_weight = n_negative / n_positive — used by XGBoost to handle class imbalance.
    Random Forest and LightGBM use class_weight="balanced" instead.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            class_weight="balanced",
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Training + evaluation helpers
# ---------------------------------------------------------------------------

def build_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """Returns (mean_auc, std_auc) from stratified 5-fold CV on the training set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def evaluate_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    target_names = ["No", "Yes"]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(report["Yes"]["precision"], 4),
        "recall":    round(report["Yes"]["recall"], 4),
        "f1":        round(report["Yes"]["f1-score"], 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    # --- Load & split ---
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = LabelEncoder().fit_transform(df[TARGET])          # No=0, Yes=1

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols     = X.select_dtypes(include="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Churn rate  — train: {y_train.mean():.1%}  |  test: {y_test.mean():.1%}\n")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight (XGBoost): {scale_pos_weight:.2f}\n")

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    models = get_models(scale_pos_weight)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline row for the comparison table
    rows = [
        {
            "Model":       "Logistic Regression (baseline)",
            "CV AUC Mean": 0.8468,   # from baseline.py run
            "CV AUC Std":  "-",
            "Test AUC":    0.8478,
            "Test F1":     0.6205,
            "Test Recall": 0.7914,
            "Train Time":  "-",
        }
    ]

    trained: dict[str, Pipeline] = {}

    for name, clf in models.items():
        print(f"{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        pipeline = build_pipeline(preprocessor, clf)

        # --- 5-fold CV on train set ---
        print(f"  Running {CV_FOLDS}-fold CV ...")
        cv_mean, cv_std = cross_validate(pipeline, X_train, y_train)
        print(f"  CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # --- Fit on full train set and time it ---
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Fit time: {elapsed:.2f}s")

        # --- Test set evaluation ---
        metrics = evaluate_test(pipeline, X_test, y_test)
        print(f"  Test AUC    : {metrics['auc_roc']:.4f}")
        print(f"  Test F1     : {metrics['f1']:.4f}  (churn class)")
        print(f"  Test Recall : {metrics['recall']:.4f}  (churn class)")
        print(f"  Test Prec.  : {metrics['precision']:.4f}  (churn class)")
        print()
        print(classification_report(
            y_test, pipeline.predict(X_test), target_names=["No", "Yes"]
        ))

        rows.append({
            "Model":       name,
            "CV AUC Mean": round(cv_mean, 4),
            "CV AUC Std":  round(cv_std, 4),
            "Test AUC":    metrics["auc_roc"],
            "Test F1":     metrics["f1"],
            "Test Recall": metrics["recall"],
            "Train Time":  f"{elapsed:.1f}s",
        })

        # --- Save ---
        slug = name.lower().replace(" ", "_")
        out_path = MODELS_DIR / f"{slug}.pkl"
        joblib.dump(pipeline, out_path)
        print(f"  Saved -> models/{slug}.pkl\n")

        trained[name] = pipeline

    # --- Comparison table ---
    table = pd.DataFrame(rows)
    print("\n" + "="*75)
    print("COMPARISON TABLE")
    print("="*75)
    print(table.to_string(index=False))

    # --- Winner ---
    # Use test AUC as tiebreaker; retention team needs recall + explainability
    candidates = table[table["Model"] != "Logistic Regression (baseline)"]
    best_row = candidates.loc[candidates["Test AUC"].idxmax()]
    print(f"\nBest model (highest Test AUC): {best_row['Model']}")
    print(
        f"  Test AUC {best_row['Test AUC']} | "
        f"F1 {best_row['Test F1']} | "
        f"Recall {best_row['Test Recall']}"
    )

    return table


if __name__ == "__main__":
    run()
