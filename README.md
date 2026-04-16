# Telco Customer Churn Prediction

Binary classification project predicting customer churn using the IBM Telco dataset.

## Exploratory Data Analysis

**Dataset:** 7,043 customers × 21 features. 3 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`), 1 binary integer (`SeniorCitizen`), and 16 categorical columns. Target: `Churn` (Yes/No).

**Key findings:**

- **Class imbalance:** 73.5% No / 26.5% Yes — warrants `class_weight="balanced"` or SMOTE during training.
- **Tenure separates classes most clearly:** churners have median tenure ~10 months vs ~38 months for retained customers; short-tenure customers are highest risk.
- **Monthly charges correlate with churn** (median $79 churned vs $65 retained), suggesting price sensitivity as a driver.
- **`TotalCharges` has 11 disguised nulls** — empty strings where `tenure = 0` (new customers with no bill). Impute to 0, do not drop.

**Modeling implications:**

- `tenure` and `TotalCharges` are highly correlated (r ≈ 0.83) — drop `TotalCharges` or apply regularisation to avoid multicollinearity.
- `MonthlyCharges` is right-skewed with a gap at the low end — consider log transform or binning before scaling.
