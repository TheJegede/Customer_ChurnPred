"""
Customer Churn Prediction — Portfolio Streamlit App
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction | Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions.csv"
CLEANED_CSV     = PROJECT_ROOT / "data" / "cleaned.csv"
RESULTS_JSON    = PROJECT_ROOT / "data" / "model_results.json"
PROD_MODEL      = PROJECT_ROOT / "models" / "production_model.pkl"
RAW_DATA_CSV    = PROJECT_ROOT / "Customer Churn Prediction" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEMO_EVAL_ROWS  = 1409

PRIMARY  = "#2563EB"
SUCCESS  = "#16A34A"
DANGER   = "#DC2626"
MUTED    = "#6B7280"
CHART_TEXT  = "#1E293B"
CHART_MUTED = "#475569"
CHART_GRID  = "#E2E8F0"
CHART_AXIS  = "#CBD5E1"

DEMO_FEATURE_IMPORTANCE = [
    {"feature": "Contract Risk Score", "importance": 0.331},
    {"feature": "Contract Month-To-Month", "importance": 0.201},
    {"feature": "Internetservice Fiber Optic", "importance": 0.036},
    {"feature": "Onlinesecurity No", "importance": 0.031},
    {"feature": "Monthly Charges Per Service", "importance": 0.026},
    {"feature": "Charges To Tenure Ratio", "importance": 0.024},
    {"feature": "Contract Two Year", "importance": 0.021},
    {"feature": "Paymentmethod Electronic Check", "importance": 0.018},
    {"feature": "Techsupport No", "importance": 0.016},
    {"feature": "Streamingmovies Yes", "importance": 0.015},
    {"feature": "Contract One Year", "importance": 0.014},
    {"feature": "Paperlessbilling No", "importance": 0.013},
    {"feature": "Paperlessbilling Yes", "importance": 0.012},
    {"feature": "Streamingmovies No", "importance": 0.011},
    {"feature": "Tenure X Contract Risk", "importance": 0.010},
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ──────────────────────────────────── */
[data-testid="stAppViewContainer"] {background: #F1F5F9;}
[data-testid="stSidebar"]          {background: #1E293B;}
[data-testid="stSidebar"] * {color: #F1F5F9 !important;}
[data-testid="stSidebar"] .stRadio label {padding: 6px 0; font-size: 0.95rem;}

/* ── Force legible text on light background ─── */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stMarkdown li, .stMarkdown strong, .stMarkdown em,
p, span, li, label, div { color: #1E293B; }

/* ── Bordered containers → card style ─────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 8px 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}

/* ── Hero ────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #0EA5E9 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 28px;
}
.hero h1 {font-size: 2.4rem; font-weight: 800; margin: 0 0 8px 0; color: white !important;}
.hero p  {font-size: 1.15rem; opacity: 0.9; margin: 0; color: white !important;}

/* ── KPI cards ───────────────────────────────── */
.kpi {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
    border-top: 4px solid var(--accent, #2563EB);
}
.kpi .value {font-size: 2rem; font-weight: 800; color: #1E293B !important; line-height: 1.1;}
.kpi .label {font-size: 0.82rem; color: #64748B !important; margin-top: 4px; font-weight: 500;
             text-transform: uppercase; letter-spacing: .05em;}
.kpi .delta {font-size: 0.8rem; margin-top: 6px; font-weight: 600; color: #475569 !important;}
.delta-pos  {color: #16A34A !important;}
.delta-neg  {color: #DC2626 !important;}

/* ── Badges ──────────────────────────────────── */
.badge {
    display: inline-block; padding: 5px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 3px;
}
.badge-blue   {background:#DBEAFE; color:#1D4ED8 !important;}
.badge-green  {background:#D1FAE5; color:#065F46 !important;}
.badge-purple {background:#EDE9FE; color:#5B21B6 !important;}
.badge-amber  {background:#FEF3C7; color:#92400E !important;}
.badge-pink   {background:#FCE7F3; color:#9D174D !important;}

/* ── Callout boxes ───────────────────────────── */
.callout {
    border-left: 4px solid #2563EB; background: #EFF6FF;
    padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 10px 0;
}
.callout-warn    {border-color: #D97706; background: #FFFBEB;}
.callout-success {border-color: #16A34A; background: #F0FDF4;}
.callout p       {margin: 0; font-size: 0.92rem; color: #1E293B !important;}
.callout strong  {color: #1E293B !important;}

/* ── Section headings ────────────────────────── */
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #1E293B !important;
    margin: 0 0 4px 0; padding: 0;
}
.section-sub {font-size: 0.88rem; color: #64748B !important; margin: 0 0 20px 0;}

/* ── Footer ──────────────────────────────────── */
.footer {
    text-align: center; font-size: 0.8rem; color: #94A3B8 !important;
    margin-top: 40px; padding: 20px 0; border-top: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
def _path_signature(path: Path) -> tuple[str, bool, int | None, int | None]:
    if path.exists():
        stat = path.stat()
        return (str(path), True, stat.st_mtime_ns, stat.st_size)
    return (str(path), False, None, None)


def _coerce_telco_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ("SeniorCitizen", "tenure"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
            errors="coerce",
        ).fillna(0.0)

    return df


def _build_demo_probability_series(df: pd.DataFrame) -> np.ndarray:
    prob = np.full(len(df), 0.08, dtype=float)

    prob += np.where(df["Contract"].eq("Month-to-month"), 0.28, 0.0)
    prob += np.where(df["Contract"].eq("One year"), 0.03, 0.0)
    prob -= np.where(df["Contract"].eq("Two year"), 0.18, 0.0)

    prob += np.where(df["tenure"] <= 12, 0.18, 0.0)
    prob += np.where((df["tenure"] > 12) & (df["tenure"] <= 24), 0.08, 0.0)
    prob -= np.where(df["tenure"] >= 48, 0.12, 0.0)

    prob += np.where(df["MonthlyCharges"] >= 80, 0.12, 0.0)
    prob += np.where((df["MonthlyCharges"] >= 65) & (df["MonthlyCharges"] < 80), 0.05, 0.0)

    prob += np.where(df["InternetService"].eq("Fiber optic"), 0.12, 0.0)
    prob -= np.where(df["InternetService"].eq("No"), 0.03, 0.0)

    prob += np.where(df["TechSupport"].eq("No"), 0.07, 0.0)
    prob += np.where(df["OnlineSecurity"].eq("No"), 0.06, 0.0)
    prob += np.where(df["PaperlessBilling"].eq("Yes"), 0.04, 0.0)
    prob += np.where(df["PaymentMethod"].eq("Electronic check"), 0.05, 0.0)
    prob += np.where(df["SeniorCitizen"].eq(1), 0.04, 0.0)

    social_ties = df["Partner"].eq("Yes") | df["Dependents"].eq("Yes")
    prob -= np.where(social_ties, 0.05, 0.0)

    return np.clip(prob, 0.03, 0.97)


def _build_demo_model_results(total_rows: int, churn_rate: float) -> dict:
    return {
        "dataset": {
            "total_rows": total_rows,
            "features_engineered": 12,
            "features_selected": 29,
            "churn_rate": round(churn_rate, 3),
        },
        "models": [
            {
                "name": "Logistic Regression",
                "label": "Baseline",
                "cv_auc": 0.8468,
                "cv_std": None,
                "test_auc": 0.8478,
                "test_f1": 0.6205,
                "test_recall": 0.7914,
                "test_precision": 0.5103,
                "test_accuracy": 0.7431,
                "train_time": "< 1s",
                "is_winner": False,
                "notes": "Demo results seeded from the portfolio narrative so the app remains runnable before training artefacts exist.",
            },
            {
                "name": "Random Forest",
                "label": "Candidate",
                "cv_auc": 0.8299,
                "cv_std": 0.0092,
                "test_auc": 0.8207,
                "test_f1": 0.5204,
                "test_recall": 0.4599,
                "test_precision": 0.5993,
                "test_accuracy": 0.7814,
                "train_time": "0.4s",
                "is_winner": False,
                "notes": "Demo benchmark retained for comparison layout.",
            },
            {
                "name": "XGBoost (default)",
                "label": "Candidate",
                "cv_auc": 0.8347,
                "cv_std": 0.0103,
                "test_auc": 0.8312,
                "test_f1": 0.6129,
                "test_recall": 0.7219,
                "test_precision": 0.5325,
                "test_accuracy": 0.7600,
                "train_time": "0.4s",
                "is_winner": False,
                "notes": "Demo benchmark retained for comparison layout.",
            },
            {
                "name": "XGBoost (Optuna tuned)",
                "label": "Winner",
                "cv_auc": 0.8498,
                "cv_std": 0.0118,
                "test_auc": 0.8481,
                "test_f1": 0.6311,
                "test_recall": 0.8075,
                "test_precision": 0.5180,
                "test_accuracy": 0.7495,
                "train_time": "0.3s",
                "is_winner": True,
                "notes": "Demo winner used until real evaluation outputs are generated.",
            },
        ],
        "winner": "XGBoost (Optuna tuned)",
        "winner_reasons": [
            "Highest recall (0.808) — catches the most churners, directly reducing revenue loss",
            "Best CV AUC (0.850) — most reliable churn-risk ranking across all decision thresholds",
            "Native feature importances — retention team can see exactly why each customer is flagged",
            "Fastest inference of all tree models (0.3 s fit) — trivial to productionise",
        ],
        "is_demo": True,
    }


def _get_demo_state() -> dict:
    missing = []
    if not CLEANED_CSV.exists():
        missing.append("cleaned dataset")
    if not PREDICTIONS_CSV.exists():
        missing.append("evaluation predictions")
    if not RESULTS_JSON.exists():
        missing.append("model results")
    if not PROD_MODEL.exists():
        missing.append("trained model")

    return {
        "using_demo": bool(missing),
        "missing": missing,
    }


CLEANED_SIG = _path_signature(CLEANED_CSV)
RAW_SIG = _path_signature(RAW_DATA_CSV)
PREDICTIONS_SIG = _path_signature(PREDICTIONS_CSV)
RESULTS_SIG = _path_signature(RESULTS_JSON)
MODEL_SIG = _path_signature(PROD_MODEL)
DEMO_STATE = _get_demo_state()


@st.cache_data
def load_full_data(cleaned_sig, raw_sig) -> pd.DataFrame:
    if CLEANED_CSV.exists():
        return _coerce_telco_df(pd.read_csv(CLEANED_CSV))
    return _coerce_telco_df(pd.read_csv(RAW_DATA_CSV))


@st.cache_data
def load_predictions(predictions_sig, cleaned_sig, raw_sig) -> pd.DataFrame:
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV)

    df = load_full_data(cleaned_sig, raw_sig)
    demo = df.sample(n=min(DEMO_EVAL_ROWS, len(df)), random_state=42).reset_index(drop=True).copy()
    probs = _build_demo_probability_series(demo)
    demo["predicted_churn"] = np.where(probs >= 0.5, "Yes", "No")
    demo["churn_probability"] = probs.round(4)
    return demo


@st.cache_data
def load_model_results(results_sig, predictions_sig, cleaned_sig, raw_sig) -> dict:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)

    df = load_full_data(cleaned_sig, raw_sig)
    churn_rate = float((df["Churn"] == "Yes").mean()) if "Churn" in df.columns else 0.265
    return _build_demo_model_results(total_rows=len(df), churn_rate=churn_rate)


@st.cache_resource
def load_model(model_sig):
    if not PROD_MODEL.exists():
        return None

    try:
        return joblib.load(PROD_MODEL)
    except Exception:
        return None


@st.cache_data
def get_global_stats(cleaned_sig, raw_sig) -> dict:
    df = load_full_data(cleaned_sig, raw_sig)
    _, bins = pd.qcut(df["MonthlyCharges"], q=4, retbins=True, duplicates="drop")
    return {
        "monthly_mean": df["MonthlyCharges"].mean(),
        "monthly_std": df["MonthlyCharges"].std() or 1.0,
        "monthly_bins": bins,
    }


def style_plotly_figure(fig):
    """Keep Plotly charts readable on the app's light dashboard background."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": CHART_TEXT},
        title_font={"color": CHART_TEXT},
        legend={
            "font": {"color": CHART_TEXT},
            "title_font": {"color": CHART_TEXT},
        },
        coloraxis_colorbar={
            "tickfont": {"color": CHART_MUTED},
            "title": {"font": {"color": CHART_TEXT}},
        },
    )
    fig.update_xaxes(
        title_font={"color": CHART_TEXT},
        tickfont={"color": CHART_MUTED},
        linecolor=CHART_AXIS,
        zerolinecolor=CHART_GRID,
    )
    fig.update_yaxes(
        title_font={"color": CHART_TEXT},
        tickfont={"color": CHART_MUTED},
        linecolor=CHART_AXIS,
        zerolinecolor=CHART_GRID,
    )
    return fig

# ---------------------------------------------------------------------------
# Feature engineering helper (mirrors engineering.py for live prediction)
# ---------------------------------------------------------------------------
def _flag(val: str) -> int:
    return 1 if str(val).strip().lower() == "yes" else 0

def engineer_input(raw: dict, stats: dict) -> pd.DataFrame:
    """Convert raw user inputs into the 29-column X the production model expects."""
    t  = raw["tenure"]
    mc = raw["MonthlyCharges"]
    tc = raw["TotalCharges"]

    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    crisk = contract_map.get(raw["Contract"], 2)

    addon_cols      = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                       "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    protection_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"]
    n_svc  = sum(_flag(raw.get(c, "No")) for c in addon_cols)
    n_prot = sum(_flag(raw.get(c, "No")) for c in protection_cols)

    zscore   = (mc - stats["monthly_mean"]) / stats["monthly_std"]
    bins     = stats["monthly_bins"]
    quartile = "Q1" if mc <= bins[1] else ("Q2" if mc <= bins[2] else ("Q3" if mc <= bins[3] else "Q4"))
    tgroup   = ("new" if t <= 12 else
                ("developing" if t <= 24 else
                 ("established" if t <= 48 else "loyal")))

    row = {
        "gender": raw["gender"], "Partner": raw["Partner"],
        "Dependents": raw["Dependents"], "PhoneService": raw["PhoneService"],
        "MultipleLines": raw["MultipleLines"], "InternetService": raw["InternetService"],
        "OnlineSecurity": raw["OnlineSecurity"], "OnlineBackup": raw["OnlineBackup"],
        "DeviceProtection": raw["DeviceProtection"], "TechSupport": raw["TechSupport"],
        "StreamingTV": raw["StreamingTV"], "StreamingMovies": raw["StreamingMovies"],
        "Contract": raw["Contract"], "PaperlessBilling": raw["PaperlessBilling"],
        "PaymentMethod": raw["PaymentMethod"],
        "tenure_group": tgroup, "monthly_charges_quartile": quartile,
        "SeniorCitizen": int(raw["SeniorCitizen"]),
        "TotalCharges": float(tc),
        "has_social_ties":          int(_flag(raw["Partner"]) or _flag(raw["Dependents"])),
        "contract_risk_score":      crisk,
        "is_fiber_optic":           int(raw["InternetService"] == "Fiber optic"),
        "num_services":             n_svc,
        "num_protection_services":  n_prot,
        "monthly_charges_zscore":   round(zscore, 4),
        "monthly_charges_per_service": round(mc / (n_svc + 1), 4),
        "tenure_x_contract_risk":   t * crisk,
        "charges_to_tenure_ratio":  round(mc / (t + 1), 4),
        "total_to_monthly_ratio":   round(tc / (mc + 1), 4),
    }
    return pd.DataFrame([row])

# ---------------------------------------------------------------------------
# Feature importance helper
# ---------------------------------------------------------------------------
@st.cache_data
def get_feature_importance(model_sig, top_n: int = 15) -> pd.DataFrame:
    model = load_model(model_sig)
    if model is None:
        return pd.DataFrame(DEMO_FEATURE_IMPORTANCE).head(top_n).copy()

    prep = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    raw_names = prep.get_feature_names_out()
    importances = clf.feature_importances_

    clean = []
    for n in raw_names:
        n = n.replace("cat__", "").replace("num__", "")
        n = n.replace("_", " ").title()
        clean.append(n)

    df = pd.DataFrame({"feature": clean, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def predict_churn_probability(raw_input: dict, stats: dict, model) -> float:
    if model is not None:
        X_pred = engineer_input(raw_input, stats)
        return float(model.predict_proba(X_pred)[0, 1])

    prob = 0.08
    prob += 0.28 if raw_input["Contract"] == "Month-to-month" else 0.03 if raw_input["Contract"] == "One year" else -0.18
    prob += 0.18 if raw_input["tenure"] <= 12 else 0.08 if raw_input["tenure"] <= 24 else -0.12 if raw_input["tenure"] >= 48 else 0.0
    prob += 0.12 if raw_input["MonthlyCharges"] >= 80 else 0.05 if raw_input["MonthlyCharges"] >= 65 else -0.02
    prob += 0.12 if raw_input["InternetService"] == "Fiber optic" else -0.03 if raw_input["InternetService"] == "No" else 0.0
    prob += 0.07 if raw_input["TechSupport"] == "No" else -0.02 if raw_input["TechSupport"] == "Yes" else 0.0
    prob += 0.06 if raw_input["OnlineSecurity"] == "No" else -0.02 if raw_input["OnlineSecurity"] == "Yes" else 0.0
    prob += 0.04 if raw_input["PaperlessBilling"] == "Yes" else 0.0
    prob += 0.05 if raw_input["PaymentMethod"] == "Electronic check" else 0.0
    prob += 0.04 if raw_input["SeniorCitizen"] else 0.0
    prob -= 0.04 if raw_input["Partner"] == "Yes" else 0.0
    prob -= 0.04 if raw_input["Dependents"] == "Yes" else 0.0
    return float(np.clip(prob, 0.03, 0.97))

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "",
        ["Project Overview", "Explore the Data", "Model Results", "How I Built This"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Taiwo Jegede**")
    st.markdown("Data Scientist")
    st.markdown("[GitHub](https://github.com/TheJegede/Customer_ChurnPred) · [LinkedIn](https://linkedin.com)")
    st.markdown("---")
    st.caption("Telco Customer Churn · 7,043 records · XGBoost · Optuna")
    if DEMO_STATE["using_demo"]:
        st.caption(f"Demo mode active: using fallback for {', '.join(DEMO_STATE['missing'])}.")

# ===========================================================================
# PAGE 1 — PROJECT OVERVIEW
# ===========================================================================
if page == "Project Overview":
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>📉 Customer Churn Prediction</h1>
        <p>End-to-end ML pipeline predicting which telecom customers will cancel —
        from raw data to a deployed, explainable model the retention team can act on.</p>
    </div>
    """, unsafe_allow_html=True)

    results = load_model_results(RESULTS_SIG, PREDICTIONS_SIG, CLEANED_SIG, RAW_SIG)
    ds      = results["dataset"]
    winner  = next(m for m in results["models"] if m["is_winner"])

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi" style="--accent:#2563EB">
            <div class="value">{ds['total_rows']:,}</div>
            <div class="label">Customers Analysed</div>
            <div class="delta">21 raw features</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi" style="--accent:#7C3AED">
            <div class="value">{ds['features_engineered']}</div>
            <div class="label">Features Engineered</div>
            <div class="delta">domain · stats · interactions</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi" style="--accent:#0891B2">
            <div class="value">{winner['test_auc']}</div>
            <div class="label">AUC-ROC (test)</div>
            <div class="delta delta-pos">Best model · XGBoost tuned</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        baseline    = next(m for m in results["models"] if m["label"] == "Baseline")
        recall_lift = round((winner["test_recall"] - baseline["test_recall"]) * 100, 1)
        st.markdown(f"""
        <div class="kpi" style="--accent:#16A34A">
            <div class="value">+{recall_lift}%</div>
            <div class="label">Recall lift vs baseline</div>
            <div class="delta delta-pos">{winner['test_recall']:.1%} vs {baseline['test_recall']:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("#### What this project does")
            st.markdown("""
This project builds a production-ready binary classifier that identifies telecom
customers at risk of cancelling their subscription. It ingests raw customer data,
applies a quality gate and cleaning pipeline, engineers 12 domain-informed features,
and tunes an XGBoost model with Optuna to maximise recall — the metric that matters
most when missing a churner costs more than a wasted retention call.

The final model catches **80.8% of churners** on held-out data, outperforming the
logistic regression baseline on every metric while remaining fully interpretable
through feature importances and an interactive prediction interface.
            """)

    with right:
        with st.container(border=True):
            st.markdown("#### Tech Stack")
            tech = {
                "Data":     [("pandas", "blue"), ("NumPy", "blue")],
                "ML":       [("scikit-learn", "purple"), ("XGBoost", "purple"), ("LightGBM", "purple")],
                "Tuning":   [("Optuna", "amber")],
                "Tracking": [("MLflow", "pink")],
                "App":      [("Streamlit", "green")],
            }
            for category, badges in tech.items():
                st.markdown(f"**{category}**")
                html = " ".join(f'<span class="badge badge-{color}">{name}</span>'
                                for name, color in badges)
                st.markdown(html, unsafe_allow_html=True)
                st.write("")

    # Pipeline overview
    with st.container(border=True):
        st.markdown("#### Pipeline at a Glance")
        steps = [
            ("📥", "Load & Profile",  "7,043 rows · 21 columns"),
            ("🔍", "Quality Gate",    "Null checks · imbalance · row count"),
            ("🧹", "Clean",           "Type coercion · imputation"),
            ("⚙️", "Engineer",        "+12 features across 3 categories"),
            ("🎯", "Select",          "Correlation + variance filter → 29 features"),
            ("🤖", "Train",           "Baseline → candidates → Optuna tuning"),
            ("📊", "Track",           "MLflow · params · metrics · artefacts"),
        ]
        cols = st.columns(len(steps))
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:12px 4px;">
                    <div style="font-size:1.8rem">{icon}</div>
                    <div style="font-weight:700; font-size:0.85rem; color:#1E293B; margin:6px 0 2px">{title}</div>
                    <div style="font-size:0.74rem; color:#64748B">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<p class="footer">Taiwo Jegede · Customer Churn Prediction Portfolio Project</p>', unsafe_allow_html=True)

# ===========================================================================
# PAGE 2 — EXPLORE THE DATA
# ===========================================================================
elif page == "Explore the Data":
    st.markdown('<p class="section-title" style="font-size:1.6rem">🔍 Explore the Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Interactive EDA on 7,043 Telco customers — filter, compare, and discover patterns.</p>', unsafe_allow_html=True)

    df = load_full_data(CLEANED_SIG, RAW_SIG)

    # ── Target distribution + Tenure histogram ───────────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        with st.container(border=True):
            st.markdown("**Target Distribution**")
            churn_counts = df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            fig_pie = px.pie(
                churn_counts, names="Churn", values="Count",
                color="Churn",
                color_discrete_map={"No": SUCCESS, "Yes": DANGER},
                hole=0.55,
            )
            style_plotly_figure(fig_pie)
            fig_pie.update_traces(
                textinfo="percent+label",
                textfont_size=13,
                textfont_color="white",
            )
            fig_pie.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=220,
                showlegend=False,
                annotations=[dict(
                    text="Churn", x=0.5, y=0.5, font_size=14,
                    font={"color": CHART_TEXT},
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True, theme=None)
            st.markdown('<div class="callout callout-warn"><p>⚠️ <strong>26.5% churn rate</strong> — class imbalance handled via <code>class_weight="balanced"</code> and XGBoost\'s <code>scale_pos_weight</code>.</p></div>', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("**Tenure Distribution by Churn Status**")
            fig_tenure = px.histogram(
                df, x="tenure", color="Churn", barmode="overlay",
                nbins=40,
                color_discrete_map={"No": SUCCESS, "Yes": DANGER},
                labels={"tenure": "Tenure (months)", "count": "Customers"},
                opacity=0.75,
            )
            style_plotly_figure(fig_tenure)
            fig_tenure.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=220,
                legend=dict(orientation="h", y=1.0, x=1, xanchor="right"),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_tenure, use_container_width=True, theme=None)
            st.markdown('<div class="callout callout-success"><p>✅ <strong>Strongest single signal:</strong> churners have median tenure ~10 months vs ~38 months for retained customers.</p></div>', unsafe_allow_html=True)

    # ── Feature explorer ─────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Feature Explorer**")
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_cols     = ["Contract", "InternetService", "PaymentMethod",
                        "TechSupport", "OnlineSecurity", "PaperlessBilling"]
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            feature_type = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True)
        with fc2:
            if feature_type == "Numeric":
                selected = st.selectbox("Choose feature", numeric_cols)
            else:
                selected = st.selectbox("Choose feature", cat_cols)

        if feature_type == "Numeric":
            fig_box = px.box(
                df, x="Churn", y=selected, color="Churn",
                color_discrete_map={"No": SUCCESS, "Yes": DANGER},
                points="outliers",
                labels={"Churn": "Churn Status"},
            )
            style_plotly_figure(fig_box)
            fig_box.update_layout(
                height=320, showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_box, use_container_width=True, theme=None)
        else:
            cat_data = (
                df.groupby([selected, "Churn"])
                  .size().reset_index(name="Count")
            )
            fig_bar = px.bar(
                cat_data, x=selected, y="Count", color="Churn",
                barmode="group",
                color_discrete_map={"No": SUCCESS, "Yes": DANGER},
            )
            style_plotly_figure(fig_bar)
            fig_bar.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    # ── Correlation heatmap ──────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Correlation Heatmap — Numeric Features**")
        num_df   = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
        corr     = num_df.corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            showscale=True,
        ))
        style_plotly_figure(fig_heat)
        corr_values = corr.round(2)
        for row_idx, y_label in enumerate(corr.index.tolist()):
            for col_idx, x_label in enumerate(corr.columns.tolist()):
                value = corr_values.iloc[row_idx, col_idx]
                fig_heat.add_annotation(
                    x=x_label,
                    y=y_label,
                    text=f"{value:g}",
                    showarrow=False,
                    font={
                        "size": 14,
                        "color": "white" if abs(value) >= 0.5 else CHART_TEXT,
                    },
                )

        fig_heat.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_heat, use_container_width=True, theme=None)
        st.markdown('<div class="callout"><p>💡 <strong>tenure ↔ TotalCharges r = 0.83</strong> — high correlation; select_features() drops TotalCharges to avoid redundancy.</p></div>', unsafe_allow_html=True)

    # ── Key findings ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title" style="margin-top:8px">Key Findings</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    findings = [
        ("📋", "Contract Type is Critical",
         "Month-to-month customers churn at <strong>3× the rate</strong> of two-year contract customers. It's the single strongest churn lever after tenure.",
         "callout"),
        ("💸", "Price Sensitivity",
         "Churners pay a median of <strong>$79/month</strong> vs $65 for retained customers. High charges with no perceived value is a primary churn driver.",
         "callout callout-warn"),
        ("🔐", "Add-ons Reduce Risk",
         "Customers with 4+ services have <strong>40% lower churn rate</strong>. Each add-on increases switching costs and perceived value.",
         "callout callout-success"),
    ]
    for col, (icon, title, body, cls) in zip([f1, f2, f3], findings):
        with col:
            st.markdown(f"""
            <div class="{cls}">
                <p><strong>{icon} {title}</strong></p>
                <p style="margin-top:6px">{body}</p>
            </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 3 — MODEL RESULTS
# ===========================================================================
elif page == "Model Results":
    st.markdown('<p class="section-title" style="font-size:1.6rem">🤖 Model Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">From baseline to production — how we selected the best model.</p>', unsafe_allow_html=True)

    results     = load_model_results(RESULTS_SIG, PREDICTIONS_SIG, CLEANED_SIG, RAW_SIG)
    models_data = results["models"]
    predictions = load_predictions(PREDICTIONS_SIG, CLEANED_SIG, RAW_SIG)
    stats       = get_global_stats(CLEANED_SIG, RAW_SIG)

    # ── Comparison table ─────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Model Comparison**")

        df_table = pd.DataFrame([{
            "Model":      m["name"],
            "Type":       m["label"],
            "CV AUC":     f'{m["cv_auc"]:.4f}',
            "Test AUC":   f'{m["test_auc"]:.4f}',
            "F1 (churn)": f'{m["test_f1"]:.4f}',
            "Recall":     f'{m["test_recall"]:.4f}',
            "Precision":  f'{m["test_precision"]:.4f}',
            "Accuracy":   f'{m["test_accuracy"]:.4f}',
            "Fit time":   m["train_time"],
        } for m in models_data])

        def highlight_winner(row):
            model_info = next((m for m in models_data if m["name"] == row["Model"]), None)
            if model_info and model_info["is_winner"]:
                return ["background-color: #F0FDF4; font-weight: 600"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_table.style.apply(highlight_winner, axis=1),
            use_container_width=True,
            hide_index=True,
            height=185,
        )

        st.markdown("**Why XGBoost (Optuna tuned)?**")
        reasons = results["winner_reasons"]
        r1, r2 = st.columns(2)
        for i, reason in enumerate(reasons):
            col = r1 if i % 2 == 0 else r2
            with col:
                st.markdown(f'<div class="callout callout-success"><p>✅ {reason}</p></div>', unsafe_allow_html=True)

    # ── Feature importance + Confusion matrix ────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Top 15 Feature Importances**")
            st.markdown('<p style="color:#64748B;font-size:0.85rem;margin:0 0 8px">What the model actually uses to predict churn</p>', unsafe_allow_html=True)
            fi_df  = get_feature_importance(MODEL_SIG, 15)
            fig_fi = px.bar(
                fi_df.sort_values("importance"),
                x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=["#BFDBFE", PRIMARY],
                labels={"importance": "Importance", "feature": ""},
            )
            style_plotly_figure(fig_fi)
            fig_fi.update_layout(
                height=420, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_fi, use_container_width=True, theme=None)
            st.markdown('<div class="callout"><p>💡 <strong>Contract Risk Score</strong> (engineered) dominates — month-to-month customers are flagged most aggressively. The model learned what the retention team already knows.</p></div>', unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown(f"**Confusion Matrix — Evaluation Set (n={len(predictions):,})**")
            actual    = (predictions["Churn"] == "Yes").astype(int)
            predicted = (predictions["predicted_churn"] == "Yes").astype(int)
            tn = int(((actual == 0) & (predicted == 0)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            tp = int(((actual == 1) & (predicted == 1)).sum())

            cm     = [[tn, fp], [fn, tp]]
            labels = ["Retained (No)", "Churn (Yes)"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0, "#F0FDF4"], [0.5, "#BFDBFE"], [1, PRIMARY]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            style_plotly_figure(fig_cm)
            annotation_text = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]
            annotation_colors = [["white", CHART_TEXT], [CHART_TEXT, "white"]]
            for row_idx, y_label in enumerate(labels):
                for col_idx, x_label in enumerate(labels):
                    fig_cm.add_annotation(
                        x=x_label,
                        y=y_label,
                        text=annotation_text[row_idx][col_idx],
                        showarrow=False,
                        font={"size": 16, "color": annotation_colors[row_idx][col_idx]},
                    )

            fig_cm.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Predicted", side="top", showgrid=False),
                yaxis=dict(title="Actual", autorange="reversed", showgrid=False),
            )
            st.plotly_chart(fig_cm, use_container_width=True, theme=None)

            recall = tp / (tp + fn)
            prec   = tp / (tp + fp)
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
                <div style="background:#F0FDF4;border-radius:8px;padding:10px;text-align:center">
                    <div style="font-size:1.4rem;font-weight:800;color:{SUCCESS}">{recall:.1%}</div>
                    <div style="font-size:0.75rem;color:#475569;font-weight:600">RECALL<br>Churners caught</div>
                </div>
                <div style="background:#EFF6FF;border-radius:8px;padding:10px;text-align:center">
                    <div style="font-size:1.4rem;font-weight:800;color:{PRIMARY}">{prec:.1%}</div>
                    <div style="font-size:0.75rem;color:#475569;font-weight:600">PRECISION<br>Flags that are real</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Try it yourself ──────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-title">🎮 Try It Yourself</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Adjust customer attributes and see the model\'s churn prediction in real time.</p>', unsafe_allow_html=True)

        model = load_model(MODEL_SIG)
        inp_col, pred_col = st.columns([2, 1])

        with inp_col:
            c1, c2, c3 = st.columns(3)
            with c1:
                tenure   = st.slider("Tenure (months)", 0, 72, 24)
                monthly  = st.slider("Monthly Charges ($)", 18.0, 118.0, 65.0, step=0.5)
                total    = st.number_input("Total Charges ($)", 0.0, 8700.0,
                                           value=float(round(tenure * monthly, 2)), step=10.0)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            with c2:
                has_internet = (internet != "No")
                no_inet_val  = "No internet service"
                tech_opt     = ["Yes", "No"] if has_internet else [no_inet_val]
                tech         = st.selectbox("Tech Support",      tech_opt)
                security     = st.selectbox("Online Security",   tech_opt if has_internet else [no_inet_val])
                backup       = st.selectbox("Online Backup",     tech_opt if has_internet else [no_inet_val])
                device       = st.selectbox("Device Protection", tech_opt if has_internet else [no_inet_val])
                streaming_tv = st.selectbox("Streaming TV",      tech_opt if has_internet else [no_inet_val])
                streaming_mo = st.selectbox("Streaming Movies",  tech_opt if has_internet else [no_inet_val])

            with c3:
                phone    = "Yes" if st.checkbox("Phone Service",   value=True)  else "No"
                lines    = st.selectbox("Multiple Lines",
                                        ["Yes", "No"] if phone == "Yes" else ["No phone service"])
                partner  = "Yes" if st.checkbox("Has Partner",     value=False) else "No"
                depends  = "Yes" if st.checkbox("Has Dependents",  value=False) else "No"
                senior   = st.checkbox("Senior Citizen",           value=False)
                paperless = "Yes" if st.checkbox("Paperless Billing", value=True) else "No"
                gender   = st.selectbox("Gender", ["Male", "Female"])
                payment  = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)",
                ])

        with pred_col:
            raw_input = {
                "gender": gender, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": depends,
                "tenure": tenure, "PhoneService": phone,
                "MultipleLines": lines, "InternetService": internet,
                "OnlineSecurity": security, "OnlineBackup": backup,
                "DeviceProtection": device, "TechSupport": tech,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_mo,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
            prob   = predict_churn_probability(raw_input, stats, model)
            label  = "Will Churn" if prob >= 0.5 else "Will Stay"
            clr    = DANGER if prob >= 0.5 else SUCCESS

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 42, "color": clr}},
                title={"text": "Churn Probability", "font": {"size": 14, "color": CHART_TEXT}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": CHART_AXIS,
                        "tickfont": {"color": CHART_MUTED},
                    },
                    "bar":  {"color": clr, "thickness": 0.25},
                    "bgcolor": "white", "borderwidth": 0,
                    "steps": [
                        {"range": [0,  33], "color": "#D1FAE5"},
                        {"range": [33, 60], "color": "#FEF3C7"},
                        {"range": [60, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": DANGER, "width": 3},
                                  "thickness": 0.75, "value": 50},
                },
            ))
            style_plotly_figure(fig_gauge)
            fig_gauge.update_layout(
                height=260, margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

            st.markdown(f"""
            <div style="text-align:center;background:{clr}15;border:2px solid {clr};
                        border-radius:10px;padding:14px;margin-top:-10px">
                <div style="font-size:1.4rem;font-weight:800;color:{clr}">{label}</div>
                <div style="font-size:0.8rem;color:#475569;margin-top:4px">
                    {'High risk — recommend intervention' if prob >= 0.5 else 'Low risk — routine engagement'}
                </div>
            </div>""", unsafe_allow_html=True)

            crisk_labels = {"Month-to-month": "🔴 High", "One year": "🟡 Medium", "Two year": "🟢 Low"}
            st.markdown(f"""
            <div style="margin-top:16px;font-size:0.82rem;color:#475569">
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #F1F5F9">
                    <span style="color:#475569">Contract risk</span>
                    <strong style="color:#1E293B">{crisk_labels.get(contract,'—')}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #F1F5F9">
                    <span style="color:#475569">Tenure stage</span>
                    <strong style="color:#1E293B">{"🔴 New" if tenure<=12 else ("🟡 Developing" if tenure<=24 else ("🟢 Established" if tenure<=48 else "🟢 Loyal"))}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0">
                    <span style="color:#475569">Internet type</span>
                    <strong style="color:#1E293B">{"⚠️ Fiber optic" if internet=="Fiber optic" else internet}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

# ===========================================================================
# PAGE 4 — HOW I BUILT THIS
# ===========================================================================
elif page == "How I Built This":
    st.markdown('<p class="section-title" style="font-size:1.6rem">🏗️ How I Built This</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Architecture, decisions, and lessons from a full ML project build.</p>', unsafe_allow_html=True)

    # ── Architecture diagram ─────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Architecture Diagram**")
        st.graphviz_chart("""
        digraph pipeline {
            rankdir=LR;
            node [shape=box style="filled,rounded" fontname="Arial" fontsize=11 width=1.4];
            edge [color="#94A3B8" penwidth=1.5];

            subgraph cluster_data {
                label="Data Layer"; style=filled; color="#EFF6FF"; fontname="Arial";
                A [label="Raw CSV\\n7,043 rows" fillcolor="#BFDBFE" color="#2563EB"];
                B [label="Quality Gate\\nnull/imbalance\\nchecks" fillcolor="#BFDBFE" color="#2563EB"];
                C [label="Cleaner\\ntype coercion\\nimputation" fillcolor="#BFDBFE" color="#2563EB"];
            }
            subgraph cluster_features {
                label="Feature Layer"; style=filled; color="#F5F3FF"; fontname="Arial";
                D [label="Feature\\nEngineering\\n+12 features" fillcolor="#DDD6FE" color="#7C3AED"];
                E [label="Feature\\nSelection\\n-2 redundant" fillcolor="#DDD6FE" color="#7C3AED"];
            }
            subgraph cluster_models {
                label="Model Layer"; style=filled; color="#FFF7ED"; fontname="Arial";
                F [label="Baseline\\nLogReg" fillcolor="#FED7AA" color="#D97706"];
                G [label="Candidates\\nRF · XGB · LGBM" fillcolor="#FED7AA" color="#D97706"];
                H [label="Optuna Tuning\\n30 trials · TPE" fillcolor="#FED7AA" color="#D97706"];
            }
            subgraph cluster_prod {
                label="Production"; style=filled; color="#F0FDF4"; fontname="Arial";
                I [label="MLflow\\nTracking" fillcolor="#BBF7D0" color="#16A34A"];
                J [label="Streamlit\\nDashboard" fillcolor="#BBF7D0" color="#16A34A"];
            }
            A -> B -> C -> D -> E;
            E -> F -> G -> H;
            H -> I -> J;
        }
        """, use_container_width=True)

    # ── Build timeline ───────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Build Timeline**")
        timeline = [
            ("Day 1", "#2563EB", "Data Loading & Quality Gate",
             "Built loader.py with PROJECT_ROOT-relative paths and a quality gate that exits 1 on critical failures (>50% nulls, <100 rows). Discovered 11 disguised nulls in TotalCharges."),
            ("Day 2", "#7C3AED", "EDA & Key Findings",
             "Explored class imbalance (73/27), identified tenure as the strongest churn signal, found MonthlyCharges/TotalCharges multicollinearity (r=0.83), and documented modeling implications."),
            ("Day 3", "#0891B2", "Feature Engineering",
             "Engineered 12 features across 3 categories: domain (contract_risk_score, has_social_ties), statistical (zscore, quartile), and interaction (charges_per_service, tenure x risk)."),
            ("Day 4", "#D97706", "Baseline Model",
             "LogisticRegression with class_weight=balanced. Surprised to find it scored AUC 0.848 — the engineered features were already linearising the signal well."),
            ("Day 5", "#DC2626", "Candidate Models",
             "Trained Random Forest, XGBoost, LightGBM. Default tree models fell below the LR baseline — Random Forest recall only 0.46. XGBoost best at 0.831 AUC."),
            ("Day 6", "#059669", "Optuna Tuning",
             "30-trial TPE search on XGBoost. Converged on shallow trees (depth=3) with slow lr (0.024). Final model beats baseline on all metrics."),
            ("Day 7", "#2563EB", "MLflow + Streamlit",
             "Logged all runs to MLflow (params + metrics + artefacts). Built this dashboard with an interactive predictor, feature importance chart, and confusion matrix."),
        ]
        for day, color, title, desc in timeline:
            t1, t2 = st.columns([1, 6])
            with t1:
                st.markdown(f"""
                <div style="text-align:center;padding-top:4px">
                    <span style="background:{color};color:white;border-radius:20px;
                                 padding:4px 12px;font-weight:700;font-size:0.82rem">{day}</span>
                </div>""", unsafe_allow_html=True)
            with t2:
                st.markdown(f"**{title}**")
                st.markdown(f'<p style="color:#475569;font-size:0.88rem;margin:2px 0 8px 0">{desc}</p>', unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid #F1F5F9;margin:4px 0">', unsafe_allow_html=True)

    # ── Key decisions ────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Key Decisions & Lessons Learned**")
        d1, d2 = st.columns(2)
        decisions = [
            ("Recall over accuracy",
             "The retention team loses more from a missed churner than from a wasted call. Using class_weight=balanced and optimising for recall was the right call — the final model catches 80.8% of churners.",
             "callout callout-success"),
            ("Engineered features beat raw trees",
             "The baseline LR outperformed default tree models. This told us the features were already linearising the signal. Shallow XGBoost (depth=3) then got the best of both worlds.",
             "callout"),
            ("Optuna over GridSearch",
             "30 TPE trials explored the space more efficiently than an exhaustive grid. Converged to a coherent config (low lr + shallow depth + high min_child_weight) in under 25 seconds.",
             "callout"),
            ("MLflow from day one",
             "Logging every run — even failed experiments — made the final comparison table trivial to produce and will be invaluable when iterating on the model post-deployment.",
             "callout callout-warn"),
        ]
        for i, (title, body, cls) in enumerate(decisions):
            col = d1 if i % 2 == 0 else d2
            with col:
                st.markdown(f'<div class="{cls}"><p><strong>{title}</strong></p><p style="margin-top:6px;font-size:0.88rem">{body}</p></div>', unsafe_allow_html=True)

    # ── GitHub link ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:12px;
                box-shadow:0 1px 3px rgba(0,0,0,.08);
                text-align:center;padding:32px;margin-top:8px">
        <div style="font-size:2rem;margin-bottom:12px">📂</div>
        <p style="font-size:1.1rem;font-weight:700;color:#1E293B;margin:0 0 6px">View the full source code</p>
        <p style="color:#64748B;font-size:0.9rem;margin:0 0 18px">
            Complete pipeline · Tests · MLflow runs · Notebooks
        </p>
        <a href="https://github.com/TheJegede/Customer_ChurnPred" target="_blank"
           style="background:#1E293B;color:white;padding:10px 28px;border-radius:8px;
                  text-decoration:none;font-weight:600;font-size:0.9rem">
            GitHub Repository ->
        </a>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="footer">Taiwo Jegede · Customer Churn Prediction Portfolio Project</p>', unsafe_allow_html=True)
