"""
Hyperparameter tuning for XGBoost using Optuna.

XGBoost was chosen as the tuning target — it was the best-performing tree model
(Test AUC 0.8312) and the one most likely to exceed the baseline (LR: 0.8478)
with proper hyperparameter search.

Search space covers:
  - Tree structure : max_depth, min_child_weight, gamma
  - Regularisation : reg_alpha (L1), reg_lambda (L2)
  - Sampling       : subsample, colsample_bytree
  - Learning       : learning_rate, n_estimators

Objective: maximise mean AUC-ROC across 5-fold stratified CV on the training set.
"""

import json
import time
import warnings
import joblib
import pandas as pd
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV  = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
PARAMS_OUT    = MODELS_DIR / "best_params.json"
MODEL_OUT     = MODELS_DIR / "tuned_model.pkl"

DROP_COLS    = ["customerID"]
TARGET       = "Churn"
N_TRIALS     = 30
CV_FOLDS     = 5
TEST_SIZE    = 0.20
RANDOM_STATE = 42

warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Suppress Optuna's per-trial INFO logs — we print our own summary instead
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_split() -> tuple:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {df.shape[0]:,} rows x {df.shape[1]} columns")

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = LabelEncoder().fit_transform(df[TARGET])   # No=0, Yes=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Churn rate — train: {y_train.mean():.1%}  |  test: {y_test.mean():.1%}\n")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessor (identical contract to train_models.py)
# ---------------------------------------------------------------------------

def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            # TotalCharges has 11 NaN rows (tenure=0) — fill 0.0 before tree split logic
            ("num", SimpleImputer(strategy="constant", fill_value=0.0), numeric_cols),
        ]
    )


def build_pipeline(preprocessor: ColumnTransformer, params: dict, scale_pos_weight: float) -> Pipeline:
    clf = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X_train, y_train, preprocessor, scale_pos_weight):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tree structure
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            # Regularisation
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # Sampling — guard against overfitting on 5k rows
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Learning rate — paired with n_estimators above
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

        pipeline = build_pipeline(preprocessor, params, scale_pos_weight)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_auc = float(scores.mean())

        # Log every trial to stdout so progress is visible
        print(
            f"  Trial {trial.number:>3} | "
            f"AUC {mean_auc:.4f} (+/- {scores.std():.4f}) | "
            f"depth={params['max_depth']} lr={params['learning_rate']:.4f} "
            f"n={params['n_estimators']}"
        )
        return mean_auc

    return objective


# ---------------------------------------------------------------------------
# Final evaluation (mirrors train_models.py evaluate_test)
# ---------------------------------------------------------------------------

def evaluate_test(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test, y_pred, target_names=["No", "Yes"], output_dict=True
    )
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

def run() -> dict:
    X_train, X_test, y_train, y_test = load_and_split()

    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numeric_cols     = X_train.select_dtypes(include="number").columns.tolist()
    preprocessor     = build_preprocessor(categorical_cols, numeric_cols)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    print(f"\nRunning Optuna TPE search — {N_TRIALS} trials, {CV_FOLDS}-fold CV ...\n")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="xgboost_churn",
    )
    objective = make_objective(X_train, y_train, preprocessor, scale_pos_weight)

    t_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_elapsed = time.time() - t_search

    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cv_auc = best_trial.value

    print(f"\nSearch complete in {search_elapsed:.1f}s")
    print(f"Best CV AUC : {best_cv_auc:.4f}  (trial #{best_trial.number})")
    print(f"Best params : {json.dumps(best_params, indent=2)}")

    # ------------------------------------------------------------------
    # Save best params
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_OUT, "w") as f:
        json.dump({"cv_auc": best_cv_auc, "params": best_params}, f, indent=2)
    print("\nBest params saved -> models/best_params.json")

    # ------------------------------------------------------------------
    # Train final model on full training set with best params
    # ------------------------------------------------------------------
    print("\nTraining final model on full training set ...")
    t_fit = time.time()
    final_pipeline = build_pipeline(preprocessor, best_params, scale_pos_weight)
    final_pipeline.fit(X_train, y_train)
    fit_elapsed = time.time() - t_fit
    print(f"Fit complete in {fit_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    metrics = evaluate_test(final_pipeline, X_test, y_test)

    print("\n" + "="*55)
    print("TUNED XGBoost — Test Set Results")
    print("="*55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (churn class)")
    print(f"  Recall    : {metrics['recall']:.4f}  (churn class)")
    print(f"  F1        : {metrics['f1']:.4f}  (churn class)")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print()
    print(classification_report(
        y_test, final_pipeline.predict(X_test), target_names=["No", "Yes"]
    ))

    # Comparison against baselines
    print("="*55)
    print("vs. baselines")
    print("="*55)
    baseline_lr  = {"auc_roc": 0.8478, "f1": 0.6205, "recall": 0.7914}
    baseline_xgb = {"auc_roc": 0.8312, "f1": 0.6129, "recall": 0.7219}
    for label, base in [("LR baseline", baseline_lr), ("Default XGBoost", baseline_xgb)]:
        auc_delta = metrics["auc_roc"] - base["auc_roc"]
        f1_delta  = metrics["f1"]      - base["f1"]
        def sign(d):
            return "+" if d >= 0 else ""
        print(
            f"  vs {label:<16} | "
            f"AUC {sign(auc_delta)}{auc_delta:+.4f} | "
            f"F1 {sign(f1_delta)}{f1_delta:+.4f}"
        )

    # ------------------------------------------------------------------
    # Save tuned model
    # ------------------------------------------------------------------
    joblib.dump(final_pipeline, MODEL_OUT)
    print("\nTuned model saved -> models/tuned_model.pkl")

    return metrics


if __name__ == "__main__":
    run()
