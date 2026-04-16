import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
MODEL_OUT = PROJECT_ROOT / "models" / "baseline.pkl"

# Columns that carry no signal or are proxies for the target
DROP_COLS = ["customerID"]
TARGET = "Churn"


def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    """
    Preprocessor + LogisticRegression in a single sklearn Pipeline.

    - Categorical columns: one-hot encoded (unknown categories ignored at inference).
    - Numeric columns: standard scaled.
    - class_weight="balanced" compensates for the 73/27 class split without SMOTE.
    - max_iter=1000 avoids convergence warnings on this dataset size.
    """
    # TotalCharges has 11 NaN rows (tenure=0 new customers) — impute to 0.0
    # before scaling, consistent with the cleaning contract in cleaner.py.
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    target_names = ["No", "Yes"]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)

    churn_metrics = report["Yes"]

    results = {
        "accuracy":  round(accuracy, 4),
        "precision": round(churn_metrics["precision"], 4),
        "recall":    round(churn_metrics["recall"], 4),
        "f1":        round(churn_metrics["f1-score"], 4),
        "auc_roc":   round(auc, 4),
    }

    print("\n--- Baseline Logistic Regression ---")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}  (churn class)")
    print(f"  Recall   : {results['recall']:.4f}  (churn class)")
    print(f"  F1       : {results['f1']:.4f}  (churn class)")
    print(f"  AUC-ROC  : {results['auc_roc']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=target_names))

    return results


def run() -> tuple[Pipeline, dict]:
    df = load_features()

    X = df.drop(columns=DROP_COLS + [TARGET])
    # Encode target: Yes=1, No=0
    y = LabelEncoder().fit_transform(df[TARGET])  # No=0, Yes=1

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    pipeline = build_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    results = evaluate(pipeline, X_test, pd.Series(y_test))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Model saved -> {MODEL_OUT.relative_to(PROJECT_ROOT)}")

    return pipeline, results


if __name__ == "__main__":
    run()
