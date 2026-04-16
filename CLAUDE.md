# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telco customer churn prediction project using the WA_Fn-UseC_-Telco-Customer-Churn dataset. The goal is a full ML pipeline from data loading through a deployed API and dashboard.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Make src/ importable as a package (run once after cloning); installs as ml_project
pip install -e .

# Run the full pipeline in order:
python src/data/loader.py                    # profile raw data
python src/data/quality.py                   # run quality gate on raw data
python src/data/cleaner.py                   # clean → data/cleaned.csv
python src/features/run_features.py          # engineer + select → data/features.csv

# Pass a custom CSV to any data script
python src/data/loader.py "path/to/file.csv"

# Run tests (tests/ is currently empty — add test files before running)
pytest tests/

# Run a single test file / test by name
pytest tests/test_loader.py
pytest tests/test_loader.py::test_function_name

# Generate app data files (run once after any model change)
python app/generate_app_data.py   # writes data/predictions.csv + data/model_results.json

# Launch the Streamlit portfolio dashboard
python -m streamlit run app/streamlit_app.py
# Then open http://localhost:8501

# MLflow — start tracking server (runs are stored in mlruns/ at project root)
# Use python -m because the mlflow executable may not be on PATH
python -m mlflow server --host 127.0.0.1 --port 5000
# Then open http://localhost:5000
# Experiment name: churn_prediction

# Re-train all models and log to MLflow
python src/models/run_training.py
```

## Architecture

```
src/          # installable package (ml_project, pip install -e .)
  data/       # loading, quality gate, cleaning
  features/   # feature engineering and selection
  models/     # baseline, candidate training, Optuna tuning, MLflow logging
app/          # streamlit_app.py (4-page portfolio dashboard) + generate_app_data.py
notebooks/    # eda.ipynb with exploratory analysis
tests/        # pytest suite (pending — only __init__.py exists)
```

`src/` is the package root (`package_dir={"": "src"}` in `setup.py`), so imports are `from data.loader import ...` — not `from src.data.loader import ...`.

**Within-package script imports:** modules in `src/data/` import each other with bare names (`from loader import load_csv`, `from quality import check_data_quality`) because they are run as scripts inside that directory. When imported from outside (e.g. tests), use the full package path: `from data.loader import load_csv`.

## Pipeline Data Flow

```
Raw CSV
  → loader.py::load_csv()          # returns DataFrame, no mutation
  → quality.py::check_data_quality()  # runs gate; exits 1 on FAIL
  → cleaner.py::clean_data()       # saves data/cleaned.csv
  → engineering.py::create_features()   # 12 new features appended
  → engineering.py::select_features()   # drops high-corr / low-variance
  → run_features.py                # saves data/features.csv
```

`data/` is gitignored — intermediate CSVs (`cleaned.csv`, `features.csv`) are never committed.

## Dataset

- **Location:** `Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 rows × 21 columns
- **Target column:** `Churn` (object: "Yes"/"No") — 73.5% No / 26.5% Yes
- **Known data quirks:**
  - `TotalCharges` is dtype `object` — 11 empty strings where `tenure = 0`; impute to 0.0, do not drop rows
  - `SeniorCitizen` is `int64` (0/1) but is a categorical feature, not a true numeric
  - `tenure` and `TotalCharges` are highly correlated (r ≈ 0.83) — `select_features()` will drop one

## Key Design Decisions

- `load_csv()` resolves paths relative to `PROJECT_ROOT` (two levels above `src/data/`), so scripts work regardless of working directory.
- `cleaner.py` constants (`NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`, `NULL_DROP_THRESHOLD`) define the cleaning contract — update these when columns change, not the functions.
- `quality.py` thresholds: null critical >50%, null warn >20%, min rows critical <100, imbalance warn <20% minority class. Quality gate is re-run after cleaning inside `clean_data()`.
- `select_features()` uses a two-pass filter: (1) drop if |corr| > 0.95 with another feature, (2) drop if variance < 1% × median variance across numeric features.
- Engineered features fall into three categories — **domain** (churn levers: `has_social_ties`, `contract_risk_score`, `is_fiber_optic`, `num_services`, `num_protection_services`), **statistical** (`tenure_group`, `monthly_charges_zscore`, `monthly_charges_quartile`), **interaction** (`monthly_charges_per_service`, `tenure_x_contract_risk`, `charges_to_tenure_ratio`, `total_to_monthly_ratio`).

## Modeling Notes (from EDA)

- Class imbalance warrants `class_weight="balanced"` or SMOTE.
- `tenure` is the strongest churn signal (median ~10 mo churned vs ~38 mo retained).
- `MonthlyCharges` is right-skewed — consider log transform or binning before scaling.
- `TotalCharges` is redundant with `tenure` + `MonthlyCharges`; `select_features()` should drop it via correlation filter.
