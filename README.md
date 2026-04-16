# Customer Churn Prediction

> End-to-end ML pipeline that identifies telecom customers at risk of churning — from raw data through feature engineering, model tuning, and an interactive Streamlit portfolio dashboard.

**[Live Demo](https://https://whychurn.streamlit.app/)** &nbsp;|&nbsp; [GitHub Actions CI](https://github.com/TheJegede/Customer_ChurnPred/actions)

---

## Project Overview

### The Problem

A telecom company loses revenue every time a customer cancels their subscription. Proactively identifying which customers are likely to churn — before they actually leave — lets the retention team target interventions (discount offers, service upgrades, proactive support) where they matter most.

### End User

A **retention analyst or business ops team** who needs a ranked list of at-risk customers each week, along with an explanation of *why* each customer is flagged, so the outreach team can tailor its approach.

### Data

The [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 raw columns covering demographics, account details, services subscribed, and monthly/total charges. Target: `Churn` (Yes/No) with a 73.5% / 26.5% class split.

### What the Model Outputs

For each customer the production pipeline outputs:

- **Binary prediction** — will churn (`1`) or will not churn (`0`)
- **Churn probability score** — a continuous risk ranking (0–1) the retention team uses to prioritise outreach
- **Feature importances** — which signals drove the prediction (surfaced in the dashboard)

### Key Design Decision

Optimise for **Recall over Precision**. The cost of missing a churner (lost lifetime revenue) is far higher than the cost of a wasted retention call. All model selection and threshold decisions reflect this priority.

---

## Architecture

```
Raw CSV (7,043 rows x 21 cols)
        |
        v
+------------------+
|   loader.py      |  load_csv() -- reads, profiles; no mutation
+--------+---------+
         |
         v
+------------------+
|   quality.py     |  check_data_quality() -- schema, null rates,
|                  |  row count, value ranges, class balance
|                  |  exits 1 on CRITICAL failure
+--------+---------+
         |
         v
+------------------+
|   cleaner.py     |  clean_data() -- coerce TotalCharges,
|                  |  drop nulls, dedup, re-run quality gate
|                  |  --> data/cleaned.csv
+--------+---------+
         |
         v
+------------------------+
|  engineering.py        |  create_features()  -- +12 features
|                        |  select_features()  -- corr + variance filter
|  run_features.py       |  --> data/features.csv (29 final features)
+----------+-------------+
           |
           v
+----------------------------------------------+
|              src/models/                     |
|                                              |
|  baseline.py     --> LogisticRegression      |
|  train_models.py --> RF . XGBoost . LightGBM |
|  tuning.py       --> Optuna TPE (30 trials)  |
|  run_training.py --> MLflow orchestrator     |
|                      --> production_model    |
+----------------------+-----------------------+
                       |
                       v
         +---------------------------+
         |  generate_app_data.py     |
         |  predictions.csv          |
         |  model_results.json       |
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         |   streamlit_app.py        |
         |   4-page dashboard        |
         |   localhost:8501          |
         +---------------------------+
```

---

## Results

All models evaluated on a held-out 20% test set (`RANDOM_STATE=42`, stratified split). Primary metric: **Recall** (catching churners). Secondary: AUC-ROC (ranking quality across all thresholds).

| Model | Test AUC | F1 | Recall | Precision | Accuracy | Notes |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.848 | 0.621 | 0.791 | 0.510 | 0.743 | Baseline — `class_weight=balanced` |
| Random Forest | 0.821 | 0.520 | 0.460 | 0.599 | 0.781 | High precision, unacceptable recall |
| XGBoost (default) | 0.831 | 0.613 | 0.722 | 0.533 | 0.760 | `scale_pos_weight=2.77` |
| **XGBoost (Optuna tuned)** | **0.848** | **0.631** | **0.808** | **0.518** | **0.750** | **Winner** |

### Improvement Over Baseline

| Metric | Baseline (LR) | Winner (Tuned XGB) | Delta |
|---|---|---|---|
| AUC-ROC | 0.848 | 0.848 | +0.000 |
| Recall | 0.791 | **0.808** | **+0.017** |
| F1 | 0.621 | **0.631** | **+0.010** |

The tuned XGBoost matches the baseline AUC while improving Recall and F1 — and crucially provides **native feature importances** that logistic regression coefficients cannot match in interpretability for a non-technical retention team.

**Why Random Forest lost:** Recall of 0.46 means it misses more than half of churners — disqualifying for a use case where catching churners is the entire point.

**Tuned hyperparameters** (Optuna TPE, 30 trials x 5-fold CV):

```
n_estimators=252  max_depth=3       learning_rate=0.024
min_child_weight=7  gamma=0.440     subsample=0.517
colsample_bytree=0.955  reg_alpha=0.0004  reg_lambda=0.030
```

Shallow trees (`max_depth=3`) prevent overfitting on ~5,600 training rows. Low learning rate with 252 estimators compensates with stability.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **pandas** | Data loading, cleaning, feature engineering |
| **scikit-learn** | Preprocessing pipelines, LogisticRegression, RandomForest, CV, metrics |
| **XGBoost** | Production model — gradient-boosted trees |
| **LightGBM** | Candidate model — histogram-based boosting |
| **Optuna** | Bayesian hyperparameter search (TPE sampler, 30 trials) |
| **MLflow** | Experiment tracking, run comparison, model artifact storage |
| **Streamlit** | Interactive 4-page portfolio dashboard |
| **Plotly** | Interactive charts in the dashboard |
| **joblib** | Model serialisation / deserialisation |
| **pytest** | Test suite — 36 tests across data quality, features, model inference |
| **ruff** | Linter — enforced in CI |
| **Docker / Compose** | Containerised app deployment |
| **GitHub Actions** | CI: pytest + ruff on every push and PR to `main` |

---

## Setup & Installation

**Prerequisites:** Python 3.9+, pip, git. Docker is optional.

```bash
# 1. Clone
git clone https://github.com/TheJegede/Customer_ChurnPred.git
cd Customer_ChurnPred

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install src/ as an editable package (required for all imports to resolve)
pip install -e .
```

The dataset is included at `Customer Churn Prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv` — no external download needed.

---

## How to Run

### Full Training Pipeline

Run these scripts in order. Each reads the output of the previous one.

```bash
# Step 1 — Profile raw data
python src/data/loader.py

# Step 2 — Quality gate (exits 1 on failure)
python src/data/quality.py

# Step 3 — Clean --> data/cleaned.csv
python src/data/cleaner.py

# Step 4 — Feature engineering + selection --> data/features.csv
python src/features/run_features.py

# Step 5 — Train all models, log to MLflow, save production_model.pkl
python src/models/run_training.py

# Step 6 — Generate dashboard artefacts (re-run after any model change)
python app/generate_app_data.py
```

### Streamlit Dashboard

```bash
python -m streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

> **Demo mode:** if `data/cleaned.csv`, `data/predictions.csv`, or `models/production_model.pkl` are missing, the dashboard automatically falls back to seeded synthetic data — the app is always runnable without running the pipeline first.

### MLflow Experiment Tracking

```bash
python -m mlflow server --host 127.0.0.1 --port 5000
# Open http://localhost:5000  |  experiment name: churn_prediction
```

### Docker

```bash
# Build and run
docker build -t churn-streamlit .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  churn-streamlit

# Or with Compose (builds + starts in one command)
docker compose up --build
```

### Tests & Lint

```bash
# Full suite (36 tests)
pytest tests/ -v

# Single file
pytest tests/test_data_quality.py -v

# Single test by name
pytest tests/test_features.py::TestFeatureRanges::test_contract_risk_score_range -v

# Lint
python -m ruff check src/ app/
python -m ruff check src/ app/ --fix    # auto-fix safe issues
```

---

## Feature Engineering

12 features were engineered across three categories before a two-pass correlation + variance filter reduced the final feature set to 29.

### Domain Features — Churn Levers

| Feature | Logic | Rationale |
|---|---|---|
| `contract_risk_score` | Month-to-month=2, One year=1, Two year=0 | Top model importance (0.331). No-penalty exit is the strongest single churn enabler. |
| `has_social_ties` | 1 if Partner=Yes OR Dependents=Yes | Shared and family plans raise switching costs — coordinating a move is harder. |
| `is_fiber_optic` | 1 if InternetService=Fiber optic | Fiber customers churn at disproportionately high rates — more tech-savvy, more competitive alternatives. |
| `num_services` | Count of 8 active add-ons | More services = deeper product lock-in and higher perceived value from the provider. |
| `num_protection_services` | Count of security, backup, device, support | These bundles are lost on switch — direct and tangible switching cost. |

### Statistical Features — Normalisation & Segmentation

| Feature | Formula | Rationale |
|---|---|---|
| `tenure_group` | Bins: new (0–12 mo), developing, established, loyal (49+ mo) | Captures lifecycle stage — new customers churn at 3x the rate of loyal ones. |
| `monthly_charges_zscore` | (x − mean) / std | Standardises MonthlyCharges for linear and distance-based models. |
| `monthly_charges_quartile` | Q1–Q4 | Non-parametric price tier; robust to the right skew in MonthlyCharges. |

### Interaction Features — Non-Linear Combinations

| Feature | Formula | Rationale |
|---|---|---|
| `monthly_charges_per_service` | MonthlyCharges / (num_services + 1) | Price-value mismatch: paying a lot for few services is a primary churn trigger. |
| `tenure_x_contract_risk` | tenure × contract_risk_score | Compounds the two strongest signals — pinpoints new customers on month-to-month contracts. |
| `charges_to_tenure_ratio` | MonthlyCharges / (tenure + 1) | New high-payers feel less valued than long-tenured customers who earned discounts. |
| `total_to_monthly_ratio` | TotalCharges / (MonthlyCharges + 1) | When far below actual tenure, signals prior plan downgrades or credits — a dissatisfaction proxy. |

---

## Key Decisions & Lessons

- **Recall as the north star prevented a costly mistake.** Optimising for accuracy would have selected Random Forest (0.781 accuracy) over the tuned XGBoost (0.750 accuracy). That choice would have *increased* churn losses because it misses 54% of churners. Defining the business metric before touching the data was the highest-leverage decision in the project.

- **Feature engineering did more than model complexity.** Logistic Regression with the 12 engineered features (AUC 0.848) nearly matched the tuned gradient-boosted model. `contract_risk_score` and the interaction terms linearised the churn signal enough that a simple model could exploit it. Domain knowledge in features often outperforms model selection.

- **Random Forest was the documented failure.** It achieved the highest accuracy and precision of any model — numbers that would look good in a summary table. Its Recall of 0.46 makes it the worst model for the actual goal. High accuracy on imbalanced data is a trap metric.

- **Optuna converged against intuition.** The search converged on `max_depth=3` despite exploring up to depth 9. On ~5,600 training rows, deep trees overfit. The result contradicted the initial assumption that more capacity would win — a concrete reminder to let the data speak through search rather than hand-tuning.

- **MLflow file-based tracking scales well for solo work.** Logging to `mlruns/` requires no running server — runs persist locally and the UI can be started at any time to compare all historical experiments. The trade-off is that it doesn't scale to a team; a remote PostgreSQL-backed tracking server would be the natural next step.

---

## File Structure

```
Customer_ChurnPred/
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                  # CI: pytest + ruff on push / PR to main
|
+-- Customer Churn Prediction/
|   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv   # source dataset (7,043 rows)
|
+-- src/                            # installable package (ml_project)
|   +-- data/
|   |   +-- loader.py               # load_csv() -- path resolution, profiling
|   |   +-- quality.py              # check_data_quality() -- 5-check gate
|   |   +-- cleaner.py              # clean_data() -- coerce, dedupe, save
|   +-- features/
|   |   +-- engineering.py          # create_features(), select_features()
|   |   +-- run_features.py         # pipeline entrypoint --> data/features.csv
|   +-- models/
|       +-- baseline.py             # LogisticRegression --> baseline.pkl
|       +-- train_models.py         # RF + XGBoost + LightGBM comparison
|       +-- tuning.py               # Optuna 30-trial TPE search
|       +-- run_training.py         # MLflow orchestrator --> production_model.pkl
|
+-- app/
|   +-- streamlit_app.py            # 4-page portfolio dashboard
|   +-- generate_app_data.py        # writes predictions.csv + model_results.json
|
+-- tests/
|   +-- test_data_quality.py        # 9 tests -- gate pass + 5 broken-data cases
|   +-- test_features.py            # 17 tests -- count, NaN, ranges, labels
|   +-- test_model.py               # 10 tests -- load, predict, proba shape
|
+-- notebooks/
|   +-- eda.ipynb                   # exploratory analysis
|
+-- models/                         # gitignored -- serialised .pkl files
|   +-- production_model.pkl        # deployed model (tuned XGBoost pipeline)
|   +-- baseline.pkl
|   +-- tuned_model.pkl
|   +-- best_params.json            # Optuna best hyperparameters
|
+-- data/                           # gitignored -- intermediate CSVs
|   +-- cleaned.csv
|   +-- features.csv
|   +-- predictions.csv
|   +-- model_results.json
|
+-- mlruns/                         # gitignored -- MLflow tracking store
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- setup.py                        # package_dir={"": "src"}
+-- CLAUDE.md                       # AI assistant guidance for this repo
```

---

## CI Status

[![CI](https://github.com/TheJegede/Customer_ChurnPred/actions/workflows/ci.yml/badge.svg)](https://github.com/TheJegede/Customer_ChurnPred/actions/workflows/ci.yml)

Two jobs run on every push and pull request to `main`:

- **Test** — `pytest tests/ -v` on Python 3.9 / ubuntu-latest (36 tests)
- **Lint** — `ruff check src/ app/`
