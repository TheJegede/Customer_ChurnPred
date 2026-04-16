"""
Centralised training + MLflow logging for all model configurations.

Loops over two configs:
  - baseline    : LogisticRegression (class_weight=balanced) — reproduces the
                  established baseline so it appears alongside tree models in the UI.
  - tuned_xgb   : XGBoost with best hyperparameters from Optuna (best_params.json).

Each run logs:
  - Params  : model name, all hyperparameters
  - Metrics : train and test accuracy, AUC-ROC, F1, precision, recall, log-loss
              (RMSE/R2 are regression metrics and do not apply to binary classification)
  - Artifact: fitted sklearn Pipeline serialised with joblib

Tracking URI is file-based (mlruns/ at project root) so runs are persisted whether or
not the MLflow server is running. Start the server afterwards to explore the UI:

    mlflow server --host 127.0.0.1 --port 5000

Then open http://localhost:5000
"""

import json
import tempfile
import warnings
import joblib
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_CSV  = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
PARAMS_JSON   = MODELS_DIR / "best_params.json"
PRODUCTION_OUT = MODELS_DIR / "production_model.pkl"

DROP_COLS    = ["customerID"]
TARGET       = "Churn"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5
EXPERIMENT   = "churn_prediction"

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------------------
# MLflow setup — file-based so runs persist without the server running
# ---------------------------------------------------------------------------

def setup_mlflow() -> None:
    tracking_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"Experiment          : {EXPERIMENT}\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_split() -> tuple:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv : {df.shape[0]:,} rows x {df.shape[1]} columns")

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = LabelEncoder().fit_transform(df[TARGET])   # No=0, Yes=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"Churn rate — train: {y_train.mean():.1%}  |  test: {y_test.mean():.1%}\n")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Shared preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    scale_numeric=True  → StandardScaler after imputation (needed for LogisticRegression).
    scale_numeric=False → impute only (tree models don't need scaling).
    """
    if scale_numeric:
        num_step = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale",  StandardScaler()),
        ])
    else:
        num_step = SimpleImputer(strategy="constant", fill_value=0.0)

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", num_step, numeric_cols),
    ])


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

def get_configs(scale_pos_weight: float) -> list[dict]:
    """
    Returns a list of model config dicts, each with:
      name       : display name and MLflow run name
      classifier : unfitted sklearn estimator
      params     : dict logged to MLflow (all hyperparameters)
      scale_num  : whether numeric features need StandardScaler
    """
    with open(PARAMS_JSON) as f:
        best = json.load(f)
    tuned_params = best["params"]

    return [
        {
            "name": "baseline_logistic_regression",
            "classifier": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model_type":    "LogisticRegression",
                "class_weight":  "balanced",
                "max_iter":      1000,
                "solver":        "lbfgs",
                "random_state":  RANDOM_STATE,
            },
            "scale_num": True,
        },
        {
            "name": "tuned_xgboost",
            "classifier": XGBClassifier(
                **tuned_params,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            ),
            "params": {
                "model_type":        "XGBClassifier",
                "scale_pos_weight":  round(scale_pos_weight, 4),
                **{k: round(v, 6) if isinstance(v, float) else v
                   for k, v in tuned_params.items()},
            },
            "scale_num": False,
        },
    ]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(pipeline: Pipeline, X, y, prefix: str) -> dict:
    """
    Compute classification metrics and return them with a train_/test_ prefix.
    Log-loss is included as a calibration signal (lower = better-calibrated probabilities).
    """
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    report = classification_report(y, y_pred, target_names=["No", "Yes"], output_dict=True)

    return {
        f"{prefix}_accuracy":  round(accuracy_score(y, y_pred), 4),
        f"{prefix}_auc_roc":   round(roc_auc_score(y, y_prob), 4),
        f"{prefix}_f1":        round(report["Yes"]["f1-score"], 4),
        f"{prefix}_precision": round(report["Yes"]["precision"], 4),
        f"{prefix}_recall":    round(report["Yes"]["recall"], 4),
        f"{prefix}_log_loss":  round(log_loss(y, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_and_log(
    config: dict,
    X_train, X_test, y_train, y_test,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[Pipeline, dict]:
    """
    Trains one model config, logs everything to MLflow, returns (pipeline, test_metrics).
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols, config["scale_num"])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   config["classifier"]),
    ])

    print(f"  Running {CV_FOLDS}-fold CV ...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    print(f"  CV AUC : {cv_mean:.4f} (+/- {cv_std:.4f})")

    print("  Fitting on full training set ...")
    pipeline.fit(X_train, y_train)

    train_metrics = compute_metrics(pipeline, X_train, y_train, "train")
    test_metrics  = compute_metrics(pipeline, X_test,  y_test,  "test")

    print(f"  Test AUC {test_metrics['test_auc_roc']:.4f} | "
          f"F1 {test_metrics['test_f1']:.4f} | "
          f"Recall {test_metrics['test_recall']:.4f}")

    # --- MLflow logging ---
    with mlflow.start_run(run_name=config["name"]):
        # Params
        mlflow.log_params(config["params"])
        mlflow.log_param("cv_folds", CV_FOLDS)

        # CV metrics
        mlflow.log_metric("cv_auc_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_auc_std",  round(cv_std, 4))

        # Train + test metrics
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)

        # Model artifact — logged via mlflow.sklearn AND saved as a joblib file
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Also attach the raw joblib file so it's downloadable from the UI
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / f"{config['name']}.pkl"
            joblib.dump(pipeline, artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="joblib")

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run ID : {run_id}")

    return pipeline, test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_and_split()

    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numeric_cols     = X_train.select_dtypes(include="number").columns.tolist()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    configs = get_configs(scale_pos_weight)

    results = []
    best_pipeline = None
    best_auc = -1.0

    for cfg in configs:
        print(f"\n{'='*55}")
        print(f"  {cfg['name']}")
        print(f"{'='*55}")

        pipeline, test_metrics = train_and_log(
            cfg, X_train, X_test, y_train, y_test,
            categorical_cols, numeric_cols,
        )

        results.append({"model": cfg["name"], **test_metrics})

        if test_metrics["test_auc_roc"] > best_auc:
            best_auc = test_metrics["test_auc_roc"]
            best_pipeline = pipeline
            best_name = cfg["name"]

    # --- Comparison table ---
    print(f"\n{'='*75}")
    print("FINAL COMPARISON")
    print(f"{'='*75}")
    df_results = pd.DataFrame(results).set_index("model")
    print(df_results[[
        "test_auc_roc", "test_f1", "test_recall", "test_precision",
        "test_accuracy", "test_log_loss"
    ]].to_string())

    # --- Save production model ---
    joblib.dump(best_pipeline, PRODUCTION_OUT)
    print(f"\nProduction model ({best_name}) saved -> models/production_model.pkl")
    print("\nTo explore all runs in the MLflow UI:")
    print("  mlflow server --host 127.0.0.1 --port 5000")
    print("  then open http://localhost:5000")


if __name__ == "__main__":
    run()
