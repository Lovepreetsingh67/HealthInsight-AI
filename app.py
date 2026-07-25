"""
HealthInsight AI - Healthcare Analysis & Risk Prediction System
------------------------------------------------------------------
A professional multi-section Streamlit dashboard that wraps a trained
RandomForestClassifier (healthcare_model.pkl) to deliver an interactive
health-risk prediction experience: landing page, two-column input form,
KPI summary cards, a colour-coded risk gauge, a personalised diet-plan
section, weekly health trend charts, a health-tips panel and a
downloadable PDF report.
"""

import os
import pickle
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthInsight AI - Healthcare Analysis & Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "healthcare_model.pkl")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "history.csv")

FEATURES = [
    "age", "bmi", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "pregnancies", "diabetes_pedigree", "cholesterol", "heart_rate",
]

FEATURE_META = {
    "age":               {"label": "Age",              "unit": "years",   "min": 1,    "max": 120,  "default": 35,   "step": 1,    "icon": "🎂",
                           "help": "The patient's age in completed years."},
    "bmi":               {"label": "BMI",               "unit": "kg/m²",   "min": 10.0, "max": 60.0, "default": 24.5, "step": 0.1,  "icon": "⚖️",
                           "help": "Body Mass Index = weight (kg) ÷ height² (m²). Healthy range is roughly 18.5–24.9; above 25 is overweight, above 30 is obese."},
    "glucose":           {"label": "Glucose",           "unit": "mg/dL",   "min": 40,   "max": 300,  "default": 110,  "step": 1,    "icon": "🍬",
                           "help": "Fasting blood glucose (blood sugar) level. Normal fasting glucose is under 100 mg/dL; 100–125 is prediabetic; 126+ suggests diabetes."},
    "blood_pressure":    {"label": "Blood Pressure",    "unit": "mmHg",    "min": 60,   "max": 200,  "default": 120,  "step": 1,    "icon": "💓",
                           "help": "Systolic blood pressure — the pressure in your arteries when your heart beats. Normal is under 120 mmHg; 130+ is considered high."},
    "skin_thickness":    {"label": "Skin Thickness",    "unit": "mm",      "min": 0.0,  "max": 99.0, "default": 25.0, "step": 0.1,  "icon": "📏",
                           "help": "Triceps skinfold thickness — measured by pinching the skin at the back of the upper arm with a skinfold caliper. Normal range is roughly 10–30 mm for adults."},
    "insulin":           {"label": "Insulin",           "unit": "mu U/mL", "min": 0,    "max": 900,  "default": 100,  "step": 1,    "icon": "💉",
                           "help": "2-hour serum insulin — measured via a blood test taken 2 hours after drinking a glucose solution (oral glucose tolerance test). Normal range is roughly 16–166 μU/mL."},
    "pregnancies":       {"label": "Pregnancies",       "unit": "count",   "min": 0,    "max": 20,   "default": 0,    "step": 1,    "icon": "🤰",
                           "help": "Total number of times the patient has been pregnant. Enter 0 if not applicable."},
    "diabetes_pedigree": {"label": "Diabetes Pedigree", "unit": "score",   "min": 0.0,  "max": 3.0,  "default": 0.5,  "step": 0.01, "icon": "🧬",
                           "help": "A score estimating diabetes risk based on family history — higher means a stronger genetic/family link. Typical average is around 0.3–0.5; most people fall between 0.08 and 2.5."},
    "cholesterol":       {"label": "Cholesterol",       "unit": "mg/dL",   "min": 80,   "max": 400,  "default": 190,  "step": 1,    "icon": "🩸",
                           "help": "Total blood cholesterol level. Below 200 mg/dL is desirable; 200–239 is borderline high; 240+ is high."},
    "heart_rate":        {"label": "Heart Rate",        "unit": "bpm",     "min": 30,   "max": 200,  "default": 75,   "step": 1,    "icon": "❤️",
                           "help": "Resting heart rate in beats per minute. A typical healthy resting rate for adults is about 60–100 bpm."},
}

# --------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);}

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(120deg, #0d9488 0%, #0891b2 45%, #6366f1 100%);
        padding: 3.2rem 2.8rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 18px 40px -12px rgba(13, 148, 136, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 260px; height: 260px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero::before {
        content: "";
        position: absolute; bottom: -90px; right: 120px;
        width: 180px; height: 180px; border-radius: 50%;
        background: rgba(255,255,255,0.06);
    }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.18);
        padding: 5px 14px; border-radius: 999px; font-size: 0.8rem;
        font-weight: 600; letter-spacing: 0.02em; margin-bottom: 0.9rem;
    }
    .hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0 0 0.5rem 0; line-height: 1.15;}
    .hero-kicker {font-size: 1rem; font-weight: 700; opacity: 0.88; margin: 0.5rem 0 0.2rem 0; letter-spacing: 0.02em;}
    .hero p {font-size: 1.05rem; opacity: 0.95; max-width: 680px; line-height: 1.6;}

    .stat-strip {display: flex; gap: 2.2rem; margin-top: 1.6rem; flex-wrap: wrap;}
    .stat-item .num {font-size: 1.5rem; font-weight: 800;}
    .stat-item .lbl {font-size: 0.8rem; opacity: 0.85;}

    /* ---------- BUTTONS ---------- */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: none;
    }
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28);}
    div[data-testid="stFormSubmitButton"] button {
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stFormSubmitButton"] button:hover {transform: translateY(-2px); box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28);}

    /* ---------- CARDS ---------- */
    .feature-card, .diet-card, .section-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #eef0f3;
        border-radius: 16px;
        padding: 1.4rem;
        height: 100%;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .feature-card:hover {transform: translateY(-5px); box-shadow: 0 14px 26px rgba(15, 23, 42, 0.09);}
    .feature-card .ico {
        width: 46px; height: 46px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; margin-bottom: 0.6rem;
        background: linear-gradient(135deg, #ccfbf1, #cffafe);
    }
    .feature-card h4 {margin: 0.3rem 0; color: #0f172a; font-size: 1.02rem;}
    .feature-card p {color: #64748b; font-size: 0.88rem; margin: 0; line-height: 1.5;}

    .section-card {
        background: #ffffff;
        border: 1px solid #eef0f3;
        border-radius: 16px;
        padding: 1.3rem 1.4rem 0.5rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .section-card h3 {margin-top: 0; color: #0f172a; font-size: 1.08rem;}

    /* ---------- KPI ---------- */
    .kpi-card {
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        color: white;
        height: 100%;
        box-shadow: 0 10px 22px -8px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {transform: translateY(-3px);}
    .kpi-card .label {font-size: 0.8rem; opacity: 0.9; font-weight: 500;}
    .kpi-card .value {font-size: 1.65rem; font-weight: 800; margin-top: 0.15rem;}

    /* ---------- TIPS ---------- */
    .tip-item {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        color: #14532d;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .tip-item.warn {background: #fffbeb; border-left-color: #f59e0b; color: #78350f;}
    .tip-item.danger {background: #fef2f2; border-left-color: #ef4444; color: #7f1d1d;}

    /* ---------- DIET PLAN ---------- */
    .diet-card {
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        border: 1px solid #eef0f3;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .diet-card:hover {box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);}
    .diet-card .diet-title {font-size: 1.02rem; font-weight: 700; color: #0f172a; margin-bottom: 0.6rem;}
    .diet-badge-danger {background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700;}
    .diet-badge-warn {background:#fef9c3; color:#854d0e; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700;}
    .diet-badge-good {background:#dcfce7; color:#166534; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700;}

    .diet-list {list-style: none; padding-left: 0; margin: 0;}
    .diet-list li {padding: 4px 0; font-size: 0.87rem; color: #334155;}
    .diet-eat li::before {content: "✔ "; color: #16a34a; font-weight: 700;}
    .diet-avoid li::before {content: "✘ "; color: #dc2626; font-weight: 700;}
    .diet-col-title {font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em; text-transform: uppercase; margin-bottom: 0.3rem;}
    .diet-col-title.eat {color: #16a34a;}
    .diet-col-title.avoid {color: #dc2626;}

    /* ---------- BADGES ---------- */
    .badge-green {background:#dcfce7; color:#166534; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.85rem;}
    .badge-yellow {background:#fef9c3; color:#854d0e; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.85rem;}
    .badge-red {background:#fee2e2; color:#991b1b; padding:5px 14px; border-radius:20px; font-weight:700; font-size:0.85rem;}

    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    hr {margin: 1.2rem 0;}

    /* ---------- ENTRANCE ANIMATION ---------- */
    @keyframes fadeSlideIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .section-card, .feature-card, .diet-card, .kpi-card, .disclaimer-banner, .calc-panel {
        animation: fadeSlideIn 0.35s ease-out;
    }

    /* ---------- KPI DELTA ---------- */
    .kpi-delta {font-size: 0.72rem; margin-top: 5px; font-weight: 700;}
    .kpi-delta.good {color: #bbf7d0;}
    .kpi-delta.bad {color: #fecaca;}
    .kpi-delta.neutral {color: #e2e8f0; opacity: 0.85;}

    /* ---------- DISCLAIMER BANNER ---------- */
    .disclaimer-banner {
        background: #fff7ed; border: 1px solid #fed7aa; border-left: 5px solid #f97316;
        color: #7c2d12; padding: 0.75rem 1.1rem; border-radius: 10px;
        font-size: 0.86rem; margin: 0.7rem 0 1.2rem 0; line-height: 1.5;
    }

    /* ---------- BMI CALCULATOR PANEL ---------- */
    .calc-panel {
        background: #f0fdfa; border: 1px dashed #5eead4; border-radius: 12px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.9rem;
    }

    /* ---------- PATIENT BADGE ---------- */
    .patient-badge {
        display: inline-block; background: #eef2ff; color: #4338ca;
        padding: 4px 13px; border-radius: 20px; font-weight: 700; font-size: 0.82rem; margin-bottom: 0.4rem;
    }

    /* ---------- MOBILE RESPONSIVENESS ---------- */
    /* The hero title/subtitle already use clamp() for fluid sizing; these
       rules handle everything else so the app stays usable on phone screens
       instead of cards overflowing or text getting cramped. */
    @media (max-width: 640px) {
        .block-container {padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.2rem;}
        .hero {padding: 1.8rem 1.4rem; border-radius: 16px;}
        .hero p {font-size: 0.92rem;}
        .stat-strip {gap: 1rem;}
        .stat-item .num {font-size: 1.2rem;}
        .stat-item .lbl {font-size: 0.72rem;}
        .feature-card, .section-card, .diet-card {padding: 1rem;}
        .kpi-card {padding: 0.9rem 1rem;}
        .kpi-card .value {font-size: 1.3rem;}
        .diet-card .diet-title {font-size: 0.95rem;}
    }
    img {max-width: 100%; height: auto;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# DARK MODE (applied conditionally based on the sidebar toggle further down —
# the widget's session_state value is already available on every rerun before
# this point runs again, so checking it here works correctly)
# --------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp {background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;}
        [data-testid="stSidebar"] {background: #111827 !important;}
        [data-testid="stSidebar"] * {color: #e2e8f0 !important;}
        .section-card, .feature-card, .diet-card {background: #1e293b !important; border-color: #334155 !important;}
        .section-card h3, .feature-card h4, .diet-card .diet-title {color: #f1f5f9 !important;}
        .feature-card p, .diet-list li {color: #94a3b8 !important;}
        h1, h2, h3, h4, h5, h6, p, label, span, div {color: #e2e8f0;}
        .stMarkdown, .stCaption {color: #cbd5e1 !important;}
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {color: #e2e8f0 !important;}
        .calc-panel {background: #1e293b !important; border-color: #0d9488 !important;}
        .disclaimer-banner {background: #422006 !important; border-color: #92400e !important; color: #fed7aa !important;}
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# DATA / MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle.get("features", FEATURES)


def load_history():
    """Not cached on purpose: history.csv is appended to on every new
    assessment, so a fresh read is needed each session to avoid showing
    stale data to a new user/browser session."""
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        if "patient_name" not in df.columns:
            df["patient_name"] = ""
        return df
    return pd.DataFrame(columns=["timestamp", "patient_name"] + FEATURES + ["risk_level", "risk_percentage"])


def ensure_history_file():
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame(columns=["timestamp", "patient_name"] + FEATURES + ["risk_level", "risk_percentage"]).to_csv(HISTORY_PATH, index=False)
        return
    # Migrate older history files that predate the patient_name column, so a
    # new row's columns don't end up misaligned with the existing header.
    existing_header = pd.read_csv(HISTORY_PATH, nrows=0)
    if "patient_name" not in existing_header.columns:
        full = pd.read_csv(HISTORY_PATH)
        full.insert(1, "patient_name", "")
        full.to_csv(HISTORY_PATH, index=False)


try:
    model, model_features = load_model()
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False
    model, model_features = None, FEATURES

history_df = load_history()

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "records" not in st.session_state:
    st.session_state.records = history_df.copy()

# --------------------------------------------------------------------------
# HELPERS — RISK
# --------------------------------------------------------------------------
def risk_band(pct):
    if pct < 33:
        return "Low Risk", "#22c55e", "badge-green"
    elif pct < 66:
        return "Moderate Risk", "#f59e0b", "badge-yellow"
    else:
        return "High Risk", "#ef4444", "badge-red"


def gauge_chart(pct):
    label, color, _ = risk_band(pct)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#dcfce7"},
                {"range": [33, 66], "color": "#fef9c3"},
                {"range": [66, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": pct},
        },
        title={"text": f"Predicted Risk — {label}", "font": {"size": 16, "color": "#334155"}},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def feature_importance_chart():
    """Global model feature importance (from the trained RandomForest) — helps
    explain, in general terms, which factors the model weighs most heavily.
    This is NOT a per-patient explanation, just overall model behavior."""
    importances = model.feature_importances_
    pairs = sorted(
        zip([FEATURE_META[f]["label"] for f in model_features], importances),
        key=lambda p: p[1],
    )
    labels = [p[0] for p in pairs]
    values = [p[1] * 100 for p in pairs]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#6366f1", text=[f"{v:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=30, t=10, b=10),
        xaxis_title="Relative Importance", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def kpi_delta_html(current, previous, unit="", lower_is_better=True, fmt="{:.1f}"):
    """Small colored delta line for a KPI card comparing to the previous assessment."""
    if previous is None:
        return ""
    diff = current - previous
    if abs(diff) < 0.05:
        return "<div class='kpi-delta neutral'>— no change since last check</div>"
    improved = (diff < 0) if lower_is_better else (diff > 0)
    arrow = "▼" if diff < 0 else "▲"
    cls = "good" if improved else "bad"
    return f"<div class='kpi-delta {cls}'>{arrow} {fmt.format(abs(diff))}{unit} vs last check</div>"


def metrics_vs_healthy_chart(inputs):
    """Horizontal bar chart comparing patient values against healthy reference limits."""
    metrics = [
        ("BMI", inputs["bmi"], 25, "kg/m²"),
        ("Glucose", inputs["glucose"], 140, "mg/dL"),
        ("Blood Pressure", inputs["blood_pressure"], 120, "mmHg"),
        ("Cholesterol", inputs["cholesterol"], 200, "mg/dL"),
    ]
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    targets = [m[2] for m in metrics]
    colors = ["#ef4444" if v > t else "#22c55e" for v, t in zip(values, targets)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h", marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside", name="Your Value",
    ))
    fig.add_trace(go.Scatter(
        y=names, x=targets, mode="markers", name="Healthy Limit",
        marker=dict(symbol="line-ns", size=22, color="#0f172a", line=dict(width=3)),
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=30, t=30, b=10),
        legend=dict(orientation="h", y=1.18, x=0),
        xaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def diet_plate_chart(labels, values, colors, title):
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent", textfont=dict(size=11),
    )])
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#334155"}},
        height=300, margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------
# HELPERS — HEALTH TIPS
# --------------------------------------------------------------------------
def generate_tips(inputs, pct):
    tips = []

    if inputs["bmi"] < 18.5:
        tips.append(("warn", "⚖️", "You are underweight. Consider a balanced, nutrient-dense diet to reach a healthy weight."))
    elif inputs["bmi"] < 25:
        tips.append(("good", "⚖️", "Your BMI is in a healthy range — keep up your current activity level."))
    elif inputs["bmi"] < 30:
        tips.append(("warn", "⚖️", "You are overweight. Increase physical activity and follow a balanced, portion-controlled diet."))
    else:
        tips.append(("danger", "⚖️", "BMI indicates obesity. A structured weight-management plan is strongly recommended."))

    if inputs["glucose"] < 100:
        tips.append(("good", "🍬", "Glucose levels look healthy — maintain a balanced, low-sugar diet."))
    elif inputs["glucose"] < 126:
        tips.append(("warn", "🍬", "Glucose is slightly elevated. Increase physical activity and reduce refined sugar intake."))
    else:
        tips.append(("danger", "🍬", "High glucose detected. Please consult a healthcare professional and follow a diabetes-friendly diet."))

    if inputs["blood_pressure"] < 120:
        tips.append(("good", "💓", "Blood pressure is in a healthy range — great work maintaining it."))
    elif inputs["blood_pressure"] < 130:
        tips.append(("warn", "💓", "Blood pressure is slightly elevated. Reduce salt intake and exercise regularly."))
    elif inputs["blood_pressure"] < 140:
        tips.append(("warn", "💓", "Stage 1 hypertension range. Monitor your blood pressure regularly and improve lifestyle habits."))
    else:
        tips.append(("danger", "💓", "High blood pressure detected. Please consult your healthcare provider soon."))

    if inputs["cholesterol"] >= 240:
        tips.append(("danger", "🩸", "Cholesterol is high. Reduce saturated fats and add more fiber and omega-3s to your diet."))
    elif inputs["cholesterol"] >= 200:
        tips.append(("warn", "🩸", "Cholesterol is borderline high. Swap red meat for lean protein and increase activity."))
    else:
        tips.append(("good", "🩸", "Cholesterol levels are healthy — continue your current habits."))

    if inputs["heart_rate"] < 60:
        tips.append(("warn", "❤️", "Resting heart rate is lower than normal. If symptomatic, consult your doctor."))
    elif inputs["heart_rate"] <= 100:
        tips.append(("good", "❤️", "Heart rate is within the normal resting range."))
    else:
        tips.append(("warn", "❤️", "Heart rate is elevated. Practice deep breathing and manage stress levels."))

    if pct >= 66:
        tips.append(("danger", "🏥", "Overall risk is high — please schedule a check-up with a healthcare professional soon."))
    elif pct >= 33:
        tips.append(("warn", "🏥", "Overall risk is moderate — small, consistent lifestyle changes can meaningfully lower it."))
    else:
        tips.append(("good", "🏥", "Overall risk is low — keep up your healthy habits and routine checkups."))

    return tips


# --------------------------------------------------------------------------
# HELPERS — DIET PLAN
# --------------------------------------------------------------------------
def generate_diet_plan(inputs):
    """Return a list of condition-specific diet cards based on abnormal metrics."""
    plans = []

    glucose = inputs["glucose"]
    if glucose >= 126:
        plans.append({
            "severity": "danger", "icon": "🍬", "title": "Diabetes-Friendly Diet Plan",
            "eat": ["Whole grains — oats, brown rice, quinoa", "Non-starchy vegetables — spinach, broccoli, beans",
                    "Lean protein — fish, chicken, tofu, lentils", "High-fiber fruits — berries, apples, pears",
                    "Nuts & seeds in moderation"],
            "avoid": ["Sugary drinks & desserts", "White bread, white rice, refined flour",
                      "Fried & heavily processed foods", "Sweetened cereals", "Excess fruit juice"],
        })
    elif glucose >= 100:
        plans.append({
            "severity": "warn", "icon": "🍬", "title": "Blood-Sugar Balancing Diet",
            "eat": ["Whole grains over refined carbs", "Fiber-rich vegetables & legumes",
                    "Lean protein at every meal", "Low-glycemic fruits — berries, citrus"],
            "avoid": ["Sugary snacks & sodas", "Large portions of white rice/pasta", "Sweetened beverages"],
        })
    else:
        plans.append({
            "severity": "good", "icon": "🍬", "title": "Balanced Blood-Sugar Maintenance",
            "eat": ["Continue a mixed diet of whole grains, vegetables & lean protein", "Stay hydrated with water over sugary drinks"],
            "avoid": ["Excess added sugar"],
        })

    bp = inputs["blood_pressure"]
    if bp >= 140:
        plans.append({
            "severity": "danger", "icon": "💓", "title": "DASH Diet for High Blood Pressure",
            "eat": ["Potassium-rich foods — bananas, spinach, sweet potato", "Low-fat dairy", "Whole grains",
                    "Fresh fruits & vegetables (6+ servings/day)"],
            "avoid": ["Table salt & salty snacks", "Processed & canned foods", "Pickles & cured meats", "Caffeine in excess"],
        })
    elif bp >= 120:
        plans.append({
            "severity": "warn", "icon": "💓", "title": "Heart-Healthy Low-Sodium Diet",
            "eat": ["Fresh vegetables & fruits", "Herbs & spices instead of salt", "Whole grains & legumes"],
            "avoid": ["Excess salt & packaged snacks", "Fast food & fried items"],
        })
    else:
        plans.append({
            "severity": "good", "icon": "💓", "title": "Heart-Healthy Maintenance Diet",
            "eat": ["Continue a low-sodium, vegetable-rich diet", "Regular hydration"],
            "avoid": ["Excess processed sodium"],
        })

    chol = inputs["cholesterol"]
    if chol >= 240:
        plans.append({
            "severity": "danger", "icon": "🩸", "title": "Cholesterol-Lowering Diet",
            "eat": ["Oats & barley (soluble fiber)", "Fatty fish — salmon, mackerel (omega-3)",
                    "Nuts — almonds, walnuts", "Olive oil instead of butter", "Fruits & vegetables"],
            "avoid": ["Fried & fast food", "Red & processed meat", "Full-fat dairy", "Trans fats / margarine"],
        })
    elif chol >= 200:
        plans.append({
            "severity": "warn", "icon": "🩸", "title": "Heart-Smart Cholesterol Diet",
            "eat": ["Lean protein & plant-based meals", "High-fiber whole grains", "Healthy fats — olive oil, avocado"],
            "avoid": ["Excess red meat", "Full-fat dairy & butter"],
        })
    else:
        plans.append({
            "severity": "good", "icon": "🩸", "title": "Healthy Cholesterol Maintenance",
            "eat": ["Continue a fiber-rich, plant-forward diet"],
            "avoid": ["Excess saturated fat"],
        })

    bmi = inputs["bmi"]
    if bmi >= 30:
        plans.append({
            "severity": "danger", "icon": "⚖️", "title": "Calorie-Controlled Weight Management Diet",
            "eat": ["High-protein, high-fiber meals to stay full", "Vegetables at every meal", "Portion-controlled whole grains"],
            "avoid": ["Sugary drinks & desserts", "Fried food & fast food", "Late-night snacking"],
        })
    elif bmi >= 25:
        plans.append({
            "severity": "warn", "icon": "⚖️", "title": "Portion-Controlled Balanced Diet",
            "eat": ["Balanced plate — half vegetables, quarter protein, quarter whole grains", "Regular meal timing"],
            "avoid": ["Oversized portions", "Frequent snacking on processed food"],
        })
    elif bmi < 18.5:
        plans.append({
            "severity": "warn", "icon": "⚖️", "title": "Nutrient-Dense Weight-Gain Diet",
            "eat": ["Calorie-dense whole foods — nuts, avocado, whole milk", "Regular protein-rich meals"],
            "avoid": ["Empty-calorie junk food as a substitute for nutrient-dense meals"],
        })
    else:
        plans.append({
            "severity": "good", "icon": "⚖️", "title": "Healthy Weight Maintenance Diet",
            "eat": ["Continue balanced meals with regular activity"],
            "avoid": ["Excess processed snacking"],
        })

    return plans


def diet_priority_plate(inputs):
    """Pick the most urgent condition and return a recommended-plate donut chart."""
    glucose, bp, chol, bmi = inputs["glucose"], inputs["blood_pressure"], inputs["cholesterol"], inputs["bmi"]

    if glucose >= 126:
        return diet_plate_chart(
            ["Non-starchy vegetables", "Lean protein", "Whole grains", "Healthy fats"],
            [50, 25, 20, 5], ["#22c55e", "#0ea5e9", "#f59e0b", "#a78bfa"],
            "🍽️ Recommended Plate — Diabetes-Friendly",
        )
    if bp >= 140:
        return diet_plate_chart(
            ["Vegetables & fruits", "Whole grains", "Low-fat dairy/protein", "Healthy fats"],
            [45, 25, 25, 5], ["#22c55e", "#f59e0b", "#0ea5e9", "#a78bfa"],
            "🍽️ Recommended Plate — DASH (Low-Sodium)",
        )
    if chol >= 240:
        return diet_plate_chart(
            ["Vegetables & fiber", "Lean/plant protein", "Whole grains", "Unsaturated fats"],
            [40, 30, 20, 10], ["#22c55e", "#0ea5e9", "#f59e0b", "#a78bfa"],
            "🍽️ Recommended Plate — Cholesterol-Friendly",
        )
    if bmi >= 30:
        return diet_plate_chart(
            ["Vegetables", "Lean protein", "Whole grains (small portion)", "Healthy fats"],
            [50, 30, 15, 5], ["#22c55e", "#0ea5e9", "#f59e0b", "#a78bfa"],
            "🍽️ Recommended Plate — Weight Management",
        )
    return diet_plate_chart(
        ["Vegetables & fruits", "Whole grains", "Protein", "Healthy fats"],
        [40, 25, 25, 10], ["#22c55e", "#0ea5e9", "#f59e0b", "#a78bfa"],
        "🍽️ Recommended Plate — Balanced Maintenance",
    )


# --------------------------------------------------------------------------
# HELPERS — PDF REPORT
# --------------------------------------------------------------------------
def _clean(text):
    replacements = {
        "⚖️": "", "🍬": "", "💓": "", "🩸": "", "❤️": "", "🏥": "", "—": "-", "→": "->",
    }
    for k, v in replacements.items():
        text = text.replace(k, "")
    # Strip any other stray non-latin-1 characters (e.g. leftover emoji) so the
    # core Helvetica font never chokes on an unsupported glyph.
    text = "".join(ch for ch in text if ord(ch) <= 255)
    return " ".join(text.split())


def _wrap_to_width(pdf, text, max_width):
    """Manually word-wrap `text` to fit within `max_width` (mm) using the
    current font metrics. Returns a list of lines. Doing this ourselves (and
    printing each line with `cell`) avoids fpdf2's `multi_cell` line-breaking
    engine entirely, which is what raises the 'not enough horizontal space'
    error on some fpdf2 versions when a page break lands mid-paragraph."""
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if pdf.get_string_width(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _ensure_space(pdf, needed_mm=12):
    """Manually trigger a page break if the next block won't fit, instead of
    relying on fpdf2's automatic mid-text page break (the trigger for the
    'not enough horizontal space' bug)."""
    if pdf.get_y() + needed_mm > pdf.page_break_trigger:
        pdf.add_page()


def _print_wrapped(pdf, text, bullet="", line_height=6.2):
    """Print `text` bullet-wrapped, one physical line per `cell` call."""
    max_width = pdf.epw - pdf.get_string_width(bullet)
    lines = _wrap_to_width(pdf, text, max_width)
    _ensure_space(pdf, line_height * len(lines))
    for i, line in enumerate(lines):
        pdf.set_x(pdf.l_margin)
        prefix = bullet if i == 0 else " " * len(bullet)
        pdf.cell(pdf.epw, line_height, f"{prefix}{line}", new_x="LMARGIN", new_y="NEXT")


def build_pdf_report(inputs, pct, label, tips, diet_plans, patient_name=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 12, "HealthInsight AI - Risk Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    if patient_name:
        pdf.cell(0, 7, _clean(f"Patient: {patient_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Prediction Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Risk Level: {label}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Risk Percentage: {pct:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Patient Inputs", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10.5)
    for key in FEATURES:
        meta = FEATURE_META[key]
        _ensure_space(pdf, 8)
        pdf.cell(0, 7, _clean(f"{meta['label']}: {inputs[key]} {meta['unit']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Health Tips", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(40, 40, 40)
    for _, _, text in tips:
        _print_wrapped(pdf, _clean(text), bullet="- ")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(15, 23, 42)
    _ensure_space(pdf, 10)
    pdf.cell(0, 8, "Diet Recommendations", new_x="LMARGIN", new_y="NEXT")
    for plan in diet_plans:
        _ensure_space(pdf, 20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        _print_wrapped(pdf, _clean(plan["title"]), bullet="- ")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        _print_wrapped(pdf, _clean("Eat: " + ", ".join(plan["eat"])), bullet="   ")
        _print_wrapped(pdf, _clean("Avoid: " + ", ".join(plan["avoid"])), bullet="   ")
        pdf.ln(1)
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(150, 150, 150)
    _print_wrapped(
        pdf,
        "This report is generated by an AI model for informational purposes only and does not "
        "substitute professional medical advice. Please consult a qualified healthcare provider.",
        line_height=5,
    )

    return bytes(pdf.output())


def predict_risk(inputs):
    X = pd.DataFrame([[inputs[f] for f in model_features]], columns=model_features)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        pct = float(proba[1]) * 100
    else:
        pct = float(model.predict(X)[0]) * 100
    return pct


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🩺 HealthInsight AI")
    st.caption("Healthcare Analysis & Risk Prediction")
    st.divider()

    nav_options = ["Home", "Risk Prediction", "Trends & History", "About"]
    nav_icons = ["house", "activity", "graph-up", "info-circle"]

    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=nav_options,
            icons=nav_icons,
            default_index=nav_options.index(st.session_state.page),
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"font-size": "15px"},
                "nav-link": {"font-size": "14.5px", "border-radius": "8px", "margin": "3px 0"},
                "nav-link-selected": {"background-color": "#0d9488"},
            },
        )
        st.session_state.page = selected
    else:
        selected = st.radio("Navigate", nav_options, index=nav_options.index(st.session_state.page))
        st.session_state.page = selected

    st.divider()
    if MODEL_LOADED:
        st.success("Model loaded ✔")
    else:
        st.error("Model failed to load")

    st.toggle("🌙 Dark Mode", key="dark_mode")

    total_assessments = len(st.session_state.records) if "records" in st.session_state else 0
    st.caption(f"📊 {total_assessments} assessment(s) recorded")
    st.caption("v2.0 · Random Forest Classifier")

page = st.session_state.page


def render_prev_next():
    """Persistent Previous / Next controls so users are never stuck on a page
    with no way back, regardless of which button brought them there."""
    idx = nav_options.index(st.session_state.page)
    st.write("")
    st.divider()
    c1, c2, c3 = st.columns([1.3, 2, 1.3])
    with c1:
        if idx > 0:
            if st.button(f"⬅  {nav_options[idx - 1]}", use_container_width=True, key=f"nav_prev_{idx}"):
                st.session_state.page = nav_options[idx - 1]
                st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center; color:#94a3b8; font-size:0.82rem; padding-top:0.5rem;'>"
            f"Step {idx + 1} of {len(nav_options)}</div>", unsafe_allow_html=True,
        )
    with c3:
        if idx < len(nav_options) - 1:
            if st.button(f"{nav_options[idx + 1]}  ➡", use_container_width=True, key=f"nav_next_{idx}"):
                st.session_state.page = nav_options[idx + 1]
                st.rerun()

# --------------------------------------------------------------------------
# HOME / LANDING PAGE
# --------------------------------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <span class="hero-badge">✨ AI-Powered Health Screening</span>
        <h1 style="font-size: clamp(1.8rem,5vw,3rem);
        font-weight:800;
        line-height:1.2;">HealthInsight AI · Healthcare Analysis &amp; Risk Prediction</h1>
        <div style="font-size:clamp(1rem,2.5vw,1.35rem);
        font-weight:600;
        margin-top:10px;">Know Your Health Risk in Under 2 Minutes</div>
        <p>HealthInsight AI analyzes your vitals and lab metrics using a trained machine
        learning model to estimate your health risk, visualize key indicators, and generate
        a personalized diet and lifestyle plan — instantly.</p>
        <div class="stat-strip">
            <div class="stat-item"><div class="num">10+</div><div class="lbl">Health Metrics Analyzed</div></div>
            <div class="stat-item"><div class="num">AI</div><div class="lbl">Random Forest Model</div></div>
            <div class="stat-item"><div class="num">&lt;2 min</div><div class="lbl">To Get Results</div></div>
            <div class="stat-item"><div class="num">PDF</div><div class="lbl">Downloadable Report</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cta1, cta2, _ = st.columns([1.1, 1.1, 2.8])
    with cta1:
        if st.button("🩺  Begin My Free Health Check", type="primary", use_container_width=True):
            st.session_state.page = "Risk Prediction"
            st.rerun()
    with cta2:
        if st.button("📊  View Sample Trends", use_container_width=True):
            st.session_state.page = "Trends & History"
            st.rerun()

    st.write("")
    st.write("")
    st.markdown("#### Why HealthInsight AI")
    c1, c2, c3, c4 = st.columns(4)
    features_info = [
        ("🧮", "AI-Powered Prediction", "Random Forest model trained on real clinical health indicators."),
        ("📊", "Visual Risk Meter", "Color-coded gauge instantly shows low, moderate or high risk."),
        ("🥗", "Personalized Diet Plans", "Condition-specific food guidance for glucose, BP, cholesterol & weight."),
        ("📄", "PDF Reports", "Download a shareable, professional PDF summary of each result."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features_info):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="ico">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.info("💡 This tool is for informational purposes only and is not a substitute for professional medical advice.")
    render_prev_next()

# --------------------------------------------------------------------------
# RISK PREDICTION PAGE
# --------------------------------------------------------------------------
elif page == "Risk Prediction":
    st.subheader("🧾 Patient Health Assessment")
    st.caption("Fill in the patient's metrics below, organized by category, then run the prediction.")

    # ---- Optional BMI calculator (kept outside the form so it reacts live) ----
    st.markdown('<div class="calc-panel">', unsafe_allow_html=True)
    use_calc = st.checkbox("🧮 Don't know your BMI? Calculate it from height & weight", key="bmi_calc_toggle")
    if use_calc:
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="height_cm")
        with hcol2:
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5, key="weight_kg")
        computed_bmi = weight_kg / ((height_cm / 100) ** 2)
        st.session_state["computed_bmi"] = round(computed_bmi, 1)
        st.success(f"📐 Calculated BMI: **{computed_bmi:.1f} kg/m²** — this has been filled into the BMI field below.")
    else:
        st.session_state.pop("computed_bmi", None)
    st.markdown('</div>', unsafe_allow_html=True)

    bmi_default = st.session_state.get("computed_bmi", FEATURE_META["bmi"]["default"])

    with st.form("prediction_form"):
        st.markdown('<div class="section-card"><h3>🩺 Patient Details</h3></div>', unsafe_allow_html=True)
        patient_name = st.text_input(
            "Patient Name / ID (optional)", placeholder="e.g. Rahul Sharma or Patient #204",
            help="Optional — lets you track this person's trends separately over time.",
        )

        st.markdown('<div class="section-card"><h3>👤 Demographics & History</h3></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(f"{FEATURE_META['age']['icon']} Age (years)", FEATURE_META["age"]["min"], FEATURE_META["age"]["max"], FEATURE_META["age"]["default"], help=FEATURE_META["age"]["help"])
            pregnancies = st.number_input(f"{FEATURE_META['pregnancies']['icon']} Pregnancies (count)", FEATURE_META["pregnancies"]["min"], FEATURE_META["pregnancies"]["max"], FEATURE_META["pregnancies"]["default"], help=FEATURE_META["pregnancies"]["help"])
        with col2:
            diabetes_pedigree = st.number_input(f"{FEATURE_META['diabetes_pedigree']['icon']} Diabetes Pedigree Score", FEATURE_META["diabetes_pedigree"]["min"], FEATURE_META["diabetes_pedigree"]["max"], FEATURE_META["diabetes_pedigree"]["default"], step=0.01, help=FEATURE_META["diabetes_pedigree"]["help"])
            bmi = st.number_input(f"{FEATURE_META['bmi']['icon']} BMI (kg/m²)", FEATURE_META["bmi"]["min"], FEATURE_META["bmi"]["max"], bmi_default, step=0.1, help=FEATURE_META["bmi"]["help"])

        st.markdown('<div class="section-card"><h3>💓 Vitals</h3></div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            blood_pressure = st.number_input(f"{FEATURE_META['blood_pressure']['icon']} Blood Pressure (mmHg)", FEATURE_META["blood_pressure"]["min"], FEATURE_META["blood_pressure"]["max"], FEATURE_META["blood_pressure"]["default"], help=FEATURE_META["blood_pressure"]["help"])
            heart_rate = st.number_input(f"{FEATURE_META['heart_rate']['icon']} Heart Rate (bpm)", FEATURE_META["heart_rate"]["min"], FEATURE_META["heart_rate"]["max"], FEATURE_META["heart_rate"]["default"], help=FEATURE_META["heart_rate"]["help"])
        with col4:
            skin_thickness = st.number_input(f"{FEATURE_META['skin_thickness']['icon']} Skin Thickness (mm)", FEATURE_META["skin_thickness"]["min"], FEATURE_META["skin_thickness"]["max"], FEATURE_META["skin_thickness"]["default"], step=0.1, help=FEATURE_META["skin_thickness"]["help"])

        st.markdown('<div class="section-card"><h3>🧪 Lab Results</h3></div>', unsafe_allow_html=True)
        col5, col6 = st.columns(2)
        with col5:
            glucose = st.number_input(f"{FEATURE_META['glucose']['icon']} Glucose (mg/dL)", FEATURE_META["glucose"]["min"], FEATURE_META["glucose"]["max"], FEATURE_META["glucose"]["default"], help=FEATURE_META["glucose"]["help"])
            insulin = st.number_input(f"{FEATURE_META['insulin']['icon']} Insulin (mu U/mL)", FEATURE_META["insulin"]["min"], FEATURE_META["insulin"]["max"], FEATURE_META["insulin"]["default"], help=FEATURE_META["insulin"]["help"])
        with col6:
            cholesterol = st.number_input(f"{FEATURE_META['cholesterol']['icon']} Cholesterol (mg/dL)", FEATURE_META["cholesterol"]["min"], FEATURE_META["cholesterol"]["max"], FEATURE_META["cholesterol"]["default"], help=FEATURE_META["cholesterol"]["help"])

        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            submitted = st.form_submit_button("🔍 Analyze Patient Health Status", type="primary", use_container_width=True)
        with btn_col2:
            reset_clicked = st.form_submit_button("♻️ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.last_result = None
        st.session_state.pop("computed_bmi", None)
        st.rerun()

    if submitted:
        inputs = {
            "age": age, "bmi": bmi, "glucose": glucose, "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness, "insulin": insulin, "pregnancies": pregnancies,
            "diabetes_pedigree": diabetes_pedigree, "cholesterol": cholesterol, "heart_rate": heart_rate,
        }
        if not MODEL_LOADED:
            st.error("The prediction model could not be loaded. Please check healthcare_model.pkl.")
        else:
            with st.spinner("🔬 Analyzing patient data..."):
                pct = predict_risk(inputs)
                label, color, badge_class = risk_band(pct)
                tips = generate_tips(inputs, pct)
                diet_plans = generate_diet_plan(inputs)

                # Find the previous assessment (for this patient if named, else the
                # last overall) BEFORE appending the new one, so KPI deltas work.
                prior_df = st.session_state.records
                if patient_name and "patient_name" in prior_df.columns:
                    prior_matches = prior_df[prior_df["patient_name"] == patient_name]
                else:
                    prior_matches = prior_df
                previous_record = prior_matches.iloc[-1].to_dict() if len(prior_matches) > 0 else None

            st.session_state.last_result = {
                "inputs": inputs, "pct": pct, "label": label, "tips": tips, "diet_plans": diet_plans,
                "patient_name": patient_name, "previous": previous_record,
            }

            new_row = pd.DataFrame([{
                "timestamp": datetime.now(),
                "patient_name": patient_name,
                **inputs,
                "risk_level": 1 if pct >= 50 else 0,
                "risk_percentage": round(pct, 1),
            }])
            st.session_state.records = pd.concat([st.session_state.records, new_row], ignore_index=True)

            # Persist to disk so history survives across sessions/restarts.
            ensure_history_file()
            new_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)

    # ---- RESULTS ----
    result = st.session_state.last_result
    if result:
        inputs, pct, label = result["inputs"], result["pct"], result["label"]
        tips, diet_plans = result["tips"], result["diet_plans"]
        patient_name = result.get("patient_name", "")
        previous = result.get("previous")
        _, color, badge_class = risk_band(pct)

        st.divider()
        if patient_name:
            st.markdown(f'<span class="patient-badge">👤 {patient_name}</span>', unsafe_allow_html=True)
        st.markdown(f"### 📋 Assessment Result &nbsp; <span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-banner">
        ⚠️ <b>Important:</b> This is an AI-generated risk estimate for informational purposes only —
        it is <b>not a medical diagnosis</b>. Please consult a qualified healthcare professional for
        clinical decisions.
        </div>
        """, unsafe_allow_html=True)

        prev_pct = previous.get("risk_percentage") if previous else None
        prev_bmi = previous.get("bmi") if previous else None
        prev_glucose = previous.get("glucose") if previous else None
        prev_bp = previous.get("blood_pressure") if previous else None

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            ("Risk Score", f"{pct:.1f}%", "linear-gradient(135deg,#0f766e,#0d9488)", kpi_delta_html(pct, prev_pct, unit="%")),
            ("BMI", f"{inputs['bmi']:.1f}", "linear-gradient(135deg,#0891b2,#0ea5e9)", kpi_delta_html(inputs["bmi"], prev_bmi)),
            ("Glucose", f"{inputs['glucose']:.0f} mg/dL", "linear-gradient(135deg,#7c3aed,#a78bfa)", kpi_delta_html(inputs["glucose"], prev_glucose, unit=" mg/dL")),
            ("Blood Pressure", f"{inputs['blood_pressure']:.0f} mmHg", "linear-gradient(135deg,#dc2626,#f87171)", kpi_delta_html(inputs["blood_pressure"], prev_bp, unit=" mmHg")),
        ]
        for col, (lbl, val, bg, delta) in zip([k1, k2, k3, k4], kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="background:{bg};">
                    <div class="label">{lbl}</div>
                    <div class="value">{val}</div>
                    {delta}
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(gauge_chart(pct), use_container_width=True)
        with g2:
            st.markdown("#### 💡 Personalized Health Tips")
            for kind, icon, text in tips:
                css = "tip-item" if kind == "good" else f"tip-item {kind}"
                st.markdown(f'<div class="{css}">{icon} {text}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 📈 Your Values vs Healthy Limits")
        st.plotly_chart(metrics_vs_healthy_chart(inputs), use_container_width=True)

        if MODEL_LOADED and hasattr(model, "feature_importances_"):
            st.write("")
            st.markdown("#### 🔍 What Influences This Prediction")
            st.caption("Overall importance the trained model places on each factor — this reflects general "
                       "model behavior, not a breakdown specific to this one result.")
            st.plotly_chart(feature_importance_chart(), use_container_width=True)

        st.divider()
        st.markdown("### 🥗 Personalized Diet Plan")
        st.caption("Diet guidance generated from your glucose, blood pressure, cholesterol and BMI results.")

        dcol1, dcol2 = st.columns([1.6, 1])
        with dcol1:
            for plan in diet_plans:
                badge_cls = f"diet-badge-{plan['severity']}"
                badge_txt = {"danger": "Needs Attention", "warn": "Watch Closely", "good": "On Track"}[plan["severity"]]
                eat_html = "".join(f"<li>{item}</li>" for item in plan["eat"])
                avoid_html = "".join(f"<li>{item}</li>" for item in plan["avoid"])
                st.markdown(f"""
                <div class="diet-card">
                    <div class="diet-title">{plan['icon']} {plan['title']} &nbsp; <span class="{badge_cls}">{badge_txt}</span></div>
                    <div style="display:flex; gap:2rem; flex-wrap:wrap;">
                        <div style="flex:1; min-width:200px;">
                            <div class="diet-col-title eat">Foods to include</div>
                            <ul class="diet-list diet-eat">{eat_html}</ul>
                        </div>
                        <div style="flex:1; min-width:200px;">
                            <div class="diet-col-title avoid">Foods to limit</div>
                            <ul class="diet-list diet-avoid">{avoid_html}</ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with dcol2:
            st.plotly_chart(diet_priority_plate(inputs), use_container_width=True)

        st.write("")
        pdf_bytes = build_pdf_report(inputs, pct, label, tips, diet_plans, patient_name=patient_name)
        st.download_button(
            "📄 Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"healthinsight_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
        )

    render_prev_next()

# --------------------------------------------------------------------------
# TRENDS & HISTORY PAGE
# --------------------------------------------------------------------------
elif page == "Trends & History":
    st.subheader("📈 Health Trends & Assessment History")

    df = st.session_state.records.copy()
    if df.empty:
        st.warning("No assessment history yet. Run a prediction from the Risk Prediction page to see trends here.")
    else:
        
        df["timestamp"] = pd.to_datetime(df["timestamp"],
        format="mixed",
        errors="coerce"
        )
        df=df.dropna(subset=["timestamp"])
        df=df.sort_values("timestamp")
        df["date"]=df["timestamp"].dt.strftime("%b %d")

        if "patient_name" in df.columns and df["patient_name"].fillna("").ne("").nunique() > 1:
            patient_names = ["All Patients"] + sorted(df["patient_name"].dropna().unique().tolist())
            chosen_patient = st.selectbox("👤 Filter by Patient", patient_names)
            if chosen_patient != "All Patients":
                df = df[df["patient_name"] == chosen_patient]

        h1, h2, h3 = st.columns(3)
        h1.metric("Total Assessments", len(df))
        h2.metric("Average Risk", f"{df['risk_percentage'].mean():.1f}%")
        h3.metric("High-Risk Cases", int((df["risk_percentage"] >= 66).sum()))

        st.write("")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("##### ⚖️ BMI Trend")
            st.line_chart(df.set_index("date")["bmi"])
        with t2:
            st.markdown("##### 🍬 Glucose Trend")
            st.line_chart(df.set_index("date")["glucose"])

        t3, t4 = st.columns(2)
        with t3:
            st.markdown("##### 💓 Blood Pressure Trend")
            st.line_chart(df.set_index("date")["blood_pressure"])
        with t4:
            st.markdown("##### 📊 Risk Percentage Trend")
            st.line_chart(df.set_index("date")["risk_percentage"])

        st.write("")
        st.markdown("#### 📊 Bar Chart View")
        st.caption("The same metrics as bar charts, for an at-a-glance comparison across assessments.")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### ⚖️ BMI by Assessment")
            st.bar_chart(df.set_index("date")["bmi"], color="#0d9488")
        with b2:
            st.markdown("##### 🍬 Glucose by Assessment")
            st.bar_chart(df.set_index("date")["glucose"], color="#f59e0b")

        b3, b4 = st.columns(2)
        with b3:
            st.markdown("##### 💓 Blood Pressure by Assessment")
            st.bar_chart(df.set_index("date")["blood_pressure"], color="#dc2626")
        with b4:
            st.markdown("##### 📊 Risk % by Assessment")
            st.bar_chart(df.set_index("date")["risk_percentage"], color="#7c3aed")

        st.write("")
        log_col1, log_col2 = st.columns([4, 1])
        with log_col1:
            st.markdown("##### 🗂️ Assessment Log")
        with log_col2:
            csv_bytes = df.drop(columns=["date"]).sort_values("timestamp", ascending=False).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export CSV", data=csv_bytes,
                file_name=f"healthinsight_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )

        log_columns = ["timestamp"]
        if "patient_name" in df.columns:
            log_columns.append("patient_name")
        log_columns += ["age", "bmi", "glucose", "blood_pressure", "cholesterol", "risk_percentage"]
        st.dataframe(
            df[log_columns].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    render_prev_next()

# --------------------------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------------------------
elif page == "About":
    st.subheader("ℹ️ About HealthInsight AI")
    st.markdown("""
    **HealthInsight AI** is a healthcare analysis and risk prediction dashboard that uses a
    trained Random Forest classifier to estimate a patient's health risk from common clinical
    indicators such as BMI, glucose, blood pressure, cholesterol and heart rate.

    **How it works**
    1. Enter patient metrics on the **Risk Prediction** page — optionally with a patient name/ID,
       and a built-in **BMI calculator** if you only know height and weight.
    2. The model returns a probability-based risk score, shown with KPI cards, a color-coded gauge,
       and — when a previous check exists — a **change indicator** on each metric.
    3. A **"What Influences This Prediction"** panel shows which factors the model weighs most heavily.
    4. A condition-specific **diet plan** is generated based on your glucose, blood pressure, cholesterol and BMI.
    5. Past assessments are tracked on the **Trends & History** page, with line charts, bar charts,
       a patient filter, and **CSV export** — data is saved to disk so it persists across sessions.
    6. Each result can be exported as a **PDF report** to share or archive.
    7. Toggle **🌙 Dark Mode** from the sidebar any time.

    **Disclaimer:** This application is intended for educational and informational purposes
    only. It does not provide medical diagnoses and should never replace consultation with a
    qualified healthcare professional.
    """)
    st.caption("HealthInsight AI · Powered by scikit-learn & Streamlit")
    render_prev_next()
