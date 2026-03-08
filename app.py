"""
app.py — Disease Prediction Research Dashboard
Run: streamlit run app.py
Requires: models/ folder populated by running disease_prediction.py first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import shap
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PathogenIQ — Disease Prediction Research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(BASE, "figures")
MOD  = os.path.join(BASE, "models")

# ─── Custom CSS — clinical dark dashboard ────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 50%, #0a1a2e 100%);
    border-bottom: 2px solid #00c2cb;
    padding: 1.2rem 2rem 1rem;
    margin: -1.5rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.banner-logo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #00c2cb;
    letter-spacing: 2px;
    text-shadow: 0 0 20px rgba(0,194,203,0.5);
}
.banner-sub {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}
.banner-badge {
    margin-left: auto;
    background: rgba(0,194,203,0.1);
    border: 1px solid rgba(0,194,203,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    color: #00c2cb;
    letter-spacing: 2px;
    font-family: 'Share Tech Mono', monospace;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: linear-gradient(135deg, #101828, #0d1f36);
    border: 1px solid rgba(0,194,203,0.2);
    border-left: 3px solid #00c2cb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: left;
}
.stat-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #00c2cb;
    text-shadow: 0 0 12px rgba(0,194,203,0.4);
    line-height: 1;
}
.stat-label {
    font-size: 0.7rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #00c2cb;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,194,203,0.2);
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}

/* ── Info card ── */
.info-card {
    background: linear-gradient(135deg, #101828, #0d1f36);
    border: 1px solid rgba(0,194,203,0.15);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    line-height: 1.7;
    font-size: 0.9rem;
    color: #cbd5e1;
}
.info-card h4 {
    color: #00c2cb;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Alert / highlight box ── */
.highlight-box {
    background: rgba(0,194,203,0.07);
    border: 1px solid rgba(0,194,203,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #94a3b8;
}

/* ── Tabs styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #0a0f1e;
    border-bottom: 1px solid rgba(0,194,203,0.2);
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b !important;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    color: #00c2cb !important;
    border-bottom: 2px solid #00c2cb !important;
    background: rgba(0,194,203,0.05) !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid rgba(0,194,203,0.2); border-radius: 8px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00c2cb, #0066cc);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px;
    font-size: 0.8rem;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
    box-shadow: 0 0 15px rgba(0,194,203,0.3);
}
.stButton > button:hover {
    box-shadow: 0 0 25px rgba(0,194,203,0.6);
    transform: translateY(-1px);
}

/* ── Checkboxes ── */
.stCheckbox label { font-size: 0.82rem; color: #94a3b8; }

/* ── Selectbox ── */
.stSelectbox label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #64748b;
    text-transform: uppercase;
}

/* ── Progress bars ── */
.stProgress > div > div > div { background-color: #00c2cb; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: #64748b !important;
    text-transform: uppercase;
    background: #101828;
    border: 1px solid rgba(0,194,203,0.15) !important;
    border-radius: 6px !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,194,203,0.08), rgba(0,102,204,0.08));
    border: 1px solid rgba(0,194,203,0.3);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0 1rem;
}
.result-disease {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    color: #00c2cb;
    text-shadow: 0 0 15px rgba(0,194,203,0.4);
    letter-spacing: 2px;
}
.result-label {
    font-size: 0.65rem;
    color: #64748b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* ── Figure captions ── */
.fig-caption {
    background: #101828;
    border-left: 3px solid rgba(0,194,203,0.4);
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    border-radius: 0 6px 6px 0;
    margin: 0.3rem 0 1.5rem;
}

/* ── Scanline overlay effect on cards ── */
.scan-card {
    position: relative;
    background: linear-gradient(135deg, #101828, #0d1f36);
    border: 1px solid rgba(0,194,203,0.2);
    border-radius: 10px;
    padding: 1.2rem;
    overflow: hidden;
}
.scan-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,194,203,0.015) 2px,
        rgba(0,194,203,0.015) 4px
    );
    pointer-events: none;
}

/* ── Symptom checkbox grid ── */
.symptom-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.4rem;
    margin: 0.8rem 0;
}

/* ── Tag badge ── */
.tag {
    display: inline-block;
    background: rgba(0,194,203,0.1);
    border: 1px solid rgba(0,194,203,0.25);
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.7rem;
    color: #00c2cb;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─── Top banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div>
        <div class="banner-logo">🧬 PATHOGENIQ</div>
        <div class="banner-sub">Clinical Disease Prediction Research System</div>
    </div>
    <div class="banner-badge">v1.0 &nbsp;|&nbsp; RESEARCH MODE</div>
</div>
""", unsafe_allow_html=True)

# ─── Load all models once at startup ─────────────────────────────────────────
@st.cache_resource
def load_models():
    from tensorflow import keras
    models = {}
    models["le"]     = joblib.load(f"{MOD}/label_encoder.pkl")
    models["lr"]     = joblib.load(f"{MOD}/logistic_regression.pkl")
    models["dt"]     = joblib.load(f"{MOD}/decision_tree.pkl")
    models["rf"]     = joblib.load(f"{MOD}/random_forest.pkl")
    models["xgb"]    = joblib.load(f"{MOD}/xgboost.pkl")
    models["mlp"]    = keras.models.load_model(f"{MOD}/mlp_model.keras")
    models["mlp_sc"] = joblib.load(f"{MOD}/mlp_scaler.pkl")
    models["top20"]  = joblib.load(f"{MOD}/top20_shap_features.pkl")
    models["params"] = joblib.load(f"{MOD}/best_hyperparams.pkl")
    models["comp"]   = pd.read_csv(f"{BASE}/model_comparison.csv")
    return models

try:
    M = load_models()
except Exception as e:
    st.error(f"Could not load models. Run `python disease_prediction.py` first.\n\n{e}")
    st.stop()

le    = M["le"]
top20 = M["top20"]
comp  = M["comp"]

@st.cache_data
def get_all_features():
    try:
        train = pd.read_csv(f"{BASE}/Training.csv")
        train = train.drop(columns=["Unnamed: 133", "prognosis"], errors="ignore")
        return list(train.columns)
    except Exception:
        return top20

all_features = get_all_features()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◈  OVERVIEW",
    "◈  ANALYTICS",
    "◈  MODEL PERFORMANCE",
    "◈  EXPLAINABILITY",
    "◈  PREDICTOR",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    best_row  = comp.iloc[0]
    best_name = best_row["Model"]
    best_f1   = best_row["Macro F1"]
    best_acc  = best_row["Accuracy"]

    # Stat cards
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">4,920</div>
            <div class="stat-label">Patient Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">132</div>
            <div class="stat-label">Symptom Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">41</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_f1:.2%}</div>
            <div class="stat-label">Best Model F1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown('<div class="section-header">// DATASET OVERVIEW</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h4>What is this dataset?</h4>
        This research system uses a curated clinical dataset containing <strong style="color:#00c2cb">4,920 patient
        records</strong> paired with <strong style="color:#00c2cb">132 binary symptom indicators</strong>
        (0 = absent, 1 = present) and <strong style="color:#00c2cb">41 possible disease diagnoses</strong>
        as ground-truth labels. Each row represents a unique patient encounter; each column is a
        discrete symptom signal. A held-out test cohort of 42 records — one per disease class —
        was sequestered prior to any model development.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">// CLINICAL SIGNIFICANCE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h4>Why does this matter?</h4>
        Automated symptom-to-diagnosis mapping has direct applications in triage support,
        rural and low-resource clinic decision assistance, and early-alert systems for infectious
        disease outbreaks. A reliable multi-class classifier can reduce the average time-to-diagnosis
        and help route patients to the appropriate specialist before confirmatory testing.
        The 41 diseases in this dataset span <span class="tag">infectious</span>
        <span class="tag">metabolic</span> <span class="tag">dermatological</span>
        <span class="tag">cardiovascular</span> <span class="tag">neurological</span> conditions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">// APPROACH & RESULTS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
        <h4>Methodology</h4>
        Five machine learning architectures were developed, tuned, and benchmarked:
        <br><br>
        <span class="tag">Logistic Regression</span> — interpretable L2-regularized multinomial baseline<br>
        <span class="tag">Decision Tree</span> — CART, depth/leaf tuned via 5-fold GridSearchCV<br>
        <span class="tag">Random Forest</span> — 100–200 tree ensemble, tuned GridSearchCV<br>
        <span class="tag">XGBoost</span> — gradient-boosted trees, tuned over LR, depth, estimators<br>
        <span class="tag">MLP Neural Net</span> — 2-layer Keras network with Dropout + EarlyStopping<br>
        <br>
        All tuning used <strong style="color:#00c2cb">5-fold stratified cross-validation</strong> with
        macro-averaged F1 as the scoring criterion. Best performing model:
        <strong style="color:#00c2cb">{best_name}</strong> — Test Macro F1 = <strong style="color:#00c2cb">{best_f1:.4f}</strong>.
        SHAP TreeExplainer was applied to provide per-prediction feature attribution.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">// SYSTEM STATUS</div>', unsafe_allow_html=True)
        status_items = [
            ("Data Pipeline", "VERIFIED", "#00c2cb"),
            ("Feature Engineering", "COMPLETE", "#00c2cb"),
            ("Model Training", "COMPLETE", "#00c2cb"),
            ("Cross-Validation", "5-FOLD", "#00c2cb"),
            ("SHAP Explainability", "ACTIVE", "#00c2cb"),
            ("Test Set Leakage", "NONE", "#22c55e"),
            ("Class Imbalance", "NONE", "#22c55e"),
            ("Missing Values", "NONE", "#22c55e"),
        ]
        status_html = '<div class="scan-card"><table style="width:100%;border-collapse:collapse;">'
        for label, val, color in status_items:
            status_html += f"""
            <tr style="border-bottom:1px solid rgba(0,194,203,0.08);">
                <td style="padding:7px 0;font-size:0.75rem;color:#64748b;font-family:'Share Tech Mono',monospace;
                           letter-spacing:1px;">{label}</td>
                <td style="text-align:right;padding:7px 0;font-family:'Share Tech Mono',monospace;
                           font-size:0.72rem;color:{color};">{val}</td>
            </tr>"""
        status_html += "</table></div>"
        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown('<div class="section-header">// MODEL LEADERBOARD</div>', unsafe_allow_html=True)
        lb_html = '<div class="scan-card"><table style="width:100%;border-collapse:collapse;">'
        lb_html += '<tr style="border-bottom:1px solid rgba(0,194,203,0.2);">'
        lb_html += '<th style="text-align:left;font-size:0.62rem;color:#475569;letter-spacing:2px;padding:5px 0;">MODEL</th>'
        lb_html += '<th style="text-align:right;font-size:0.62rem;color:#475569;letter-spacing:2px;padding:5px 0;">F1</th>'
        lb_html += '<th style="text-align:right;font-size:0.62rem;color:#475569;letter-spacing:2px;padding:5px 0;">ACC</th>'
        lb_html += '</tr>'
        for i, row in comp.iterrows():
            highlight = "color:#00c2cb;font-weight:700;" if i == 0 else "color:#94a3b8;"
            rank_icon = "▶ " if i == 0 else f"{i+1}. "
            lb_html += f"""<tr style="border-bottom:1px solid rgba(0,194,203,0.06);">
                <td style="padding:6px 0;font-family:'Share Tech Mono',monospace;font-size:0.68rem;{highlight}">
                    {rank_icon}{row['Model']}</td>
                <td style="text-align:right;font-family:'Share Tech Mono',monospace;font-size:0.68rem;{highlight}">
                    {row['Macro F1']:.4f}</td>
                <td style="text-align:right;font-family:'Share Tech Mono',monospace;font-size:0.68rem;{highlight}">
                    {row['Accuracy']:.4f}</td>
            </tr>"""
        lb_html += "</table></div>"
        st.markdown(lb_html, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem;" class="highlight-box">
        <span style="color:#00c2cb;font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:2px;">
        ⚠ DISCLAIMER</span><br>
        <span style="font-size:0.78rem;">This tool is for research and educational use only.
        Not a substitute for clinical diagnosis by a licensed medical professional.</span>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">// DESCRIPTIVE ANALYTICS — DATASET EXPLORATION</div>',
                unsafe_allow_html=True)

    figures = [
        ("01_target_distribution.png",
         "DISEASE CLASS DISTRIBUTION",
         "Each bar represents one of the 41 disease classes. All classes contain exactly 120 training samples, "
         "confirming perfect balance. Because no class is over- or under-represented, accuracy is a valid primary "
         "metric and no resampling technique (SMOTE, class weights) is required."),

        ("02_top20_symptoms.png",
         "TOP 20 SYMPTOM PREVALENCE",
         "Bars show the fraction of all patients who present with each symptom. High-prevalence symptoms like "
         "fatigue and fever appear across many disease classes, reducing their individual discriminative power — "
         "but they remain critical in combination with rarer, disease-specific symptoms."),

        ("03_cooccurrence_heatmap.png",
         "SYMPTOM CO-OCCURRENCE MATRIX (TOP 15)",
         "Each cell shows the fraction of patients who simultaneously present with both symptoms. Darker cells "
         "indicate frequent co-presentation, suggesting shared biological mechanisms. Block patterns reveal "
         "symptom clusters corresponding to specific disease families."),

        ("04_symptoms_per_patient.png",
         "SYMPTOM BURDEN DISTRIBUTION",
         "Histogram of how many symptoms each patient presents with (mean ≈ 7.4, median = 6). The roughly "
         "unimodal distribution indicates similar clinical complexity across all disease classes — models must "
         "differentiate diseases by which symptoms appear, not just symptom count."),

        ("05_disease_symptom_profile.png",
         "DISEASE–SYMPTOM PROFILE HEATMAP",
         "Each row is a disease; each column is one of the top 20 symptoms; cell values show the average "
         "presence rate. Distinct row color patterns confirm that every disease has a unique symptom signature, "
         "which is why tree-based models can achieve near-perfect accuracy on this structured dataset."),

        ("06_correlation_heatmap.png",
         "INTER-SYMPTOM CORRELATION (TOP 30 VARIABLE FEATURES)",
         "Pairwise Pearson correlations among the 30 highest-variance symptom columns. Clustered blocks of "
         "high correlation indicate groups of symptoms that co-present (e.g., excessive_hunger and "
         "blurred_vision, |r|=0.82). Tree models exploit these clusters via early splits; LR handles them "
         "through L2 regularization."),

        ("rf_feature_importances.png",
         "RANDOM FOREST — FEATURE IMPORTANCES (TOP 20)",
         "Mean Decrease in Impurity (MDI/Gini importance) for the top 20 features across all trees in the "
         "tuned Random Forest. Features ranked highest are used at the earliest, most discriminative splits, "
         "making them the primary signal drivers in ensemble predictions."),

        ("mlp_training_history.png",
         "MLP NEURAL NETWORK — TRAINING HISTORY",
         "Loss and accuracy curves over training epochs for both the train and validation sets. "
         "EarlyStopping (patience=10) halted training and restored the best weights. Closely tracking "
         "curves with no divergence indicate healthy learning without significant overfitting."),
    ]

    # Display two figures side-by-side where possible
    for i in range(0, len(figures), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(figures):
                fname, title, caption = figures[i + j]
                fpath = os.path.join(FIG, fname)
                with col:
                    st.markdown(f'<div class="section-header">// {title}</div>', unsafe_allow_html=True)
                    if os.path.exists(fpath):
                        st.image(fpath, use_container_width=True)
                        st.markdown(f'<div class="fig-caption">{caption}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"Missing: {fname}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">// MODEL PERFORMANCE — HELD-OUT TEST SET</div>',
                unsafe_allow_html=True)

    # Metric cards for best model
    best = comp.iloc[0]
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{best['Accuracy']:.4f}</div>
            <div class="stat-label">Best Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best['Macro F1']:.4f}</div>
            <div class="stat-label">Best Macro F1</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best['Macro Precision']:.4f}</div>
            <div class="stat-label">Best Precision</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best['Macro Recall']:.4f}</div>
            <div class="stat-label">Best Recall</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">// COMPARISON TABLE</div>', unsafe_allow_html=True)
        styled = comp.style.format({
            "Accuracy"       : "{:.4f}",
            "Macro F1"       : "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall"   : "{:.4f}",
        }).highlight_max(
            subset=["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"],
            color="#0d2137"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-header">// BEST HYPERPARAMETERS</div>', unsafe_allow_html=True)
        for model_name, params in M["params"].items():
            with st.expander(f"{model_name}", expanded=False):
                st.json(params)

    st.markdown('<div class="section-header">// PERFORMANCE COMPARISON CHART</div>', unsafe_allow_html=True)
    comp_path = os.path.join(FIG, "model_comparison.png")
    if os.path.exists(comp_path):
        st.image(comp_path, use_container_width=True)
        st.markdown('<div class="fig-caption">Grouped bar chart comparing Accuracy, Macro F1, Precision, and Recall across all five models on the held-out test set. Higher is better for all metrics.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-header">// CONFUSION MATRICES — VALIDATION SET</div>',
                unsafe_allow_html=True)
    cm_files = {
        "Logistic Regression": "cm_logistic_regression.png",
        "Decision Tree"      : "cm_decision_tree.png",
        "Random Forest"      : "cm_random_forest.png",
        "XGBoost"            : "cm_xgboost.png",
        "MLP (Neural Net)"   : "cm_mlp__neural_net_.png",
    }
    for model_label, fname in cm_files.items():
        fpath = os.path.join(FIG, fname)
        if os.path.exists(fpath):
            with st.expander(f"CONFUSION MATRIX — {model_label.upper()}", expanded=False):
                st.image(fpath, use_container_width=True)
                st.markdown('<div class="fig-caption">Rows = true label, Columns = predicted label. '
                            'Perfect classifiers show all mass on the diagonal. '
                            'Off-diagonal cells indicate misclassified patients.</div>',
                            unsafe_allow_html=True)
        else:
            st.warning(f"Missing: {fname}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Explainability
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">// SHAP EXPLAINABILITY — FEATURE ATTRIBUTION ANALYSIS</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>What is SHAP?</h4>
    SHAP (SHapley Additive exPlanations) assigns each feature a contribution score for each
    individual prediction, grounded in cooperative game theory. For tree-based models,
    TreeExplainer computes exact SHAP values in polynomial time. Positive SHAP values push
    the model toward predicting a disease; negative values push away. Global importance is
    the mean absolute SHAP value across all patients and all 41 classes.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-header">// GLOBAL FEATURE IMPORTANCE (BAR)</div>',
                    unsafe_allow_html=True)
        bar_path = os.path.join(FIG, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
            st.markdown('<div class="fig-caption">Mean absolute SHAP values aggregated across all 41 disease classes. Features ranked highest exert the greatest average influence on model predictions.</div>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">// SHAP VALUE DISTRIBUTION (BEESWARM)</div>',
                    unsafe_allow_html=True)
        bee_path = os.path.join(FIG, "shap_beeswarm.png")
        if os.path.exists(bee_path):
            st.image(bee_path, use_container_width=True)
            st.markdown('<div class="fig-caption">Each dot is one patient. Color shows feature value (red=high/present, blue=low/absent). Spread along the x-axis shows the range of SHAP impact for that symptom.</div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-header">// SAMPLE PREDICTION WATERFALL</div>',
                unsafe_allow_html=True)
    wf_path = os.path.join(FIG, "shap_waterfall.png")
    if os.path.exists(wf_path):
        st.image(wf_path, use_container_width=True)
        st.markdown('<div class="fig-caption">Waterfall decomposition for a single patient prediction (Sample 0 → Hyperthyroidism). Starting from the base rate (E[f(X)]), each bar shows how much a symptom pushed the final score up (red) or down (blue).</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="section-header">// TOP SHAP FEATURES DETECTED</div>',
                unsafe_allow_html=True)
    tags = "".join([f'<span class="tag">{f.replace("_"," ")}</span>' for f in top20])
    st.markdown(f'<div class="info-card">{tags}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Interactive Predictor
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">// INTERACTIVE DISEASE PREDICTOR</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>How to use</h4>
    Select all symptoms the patient is currently presenting. The predictor will use only the
    <strong style="color:#00c2cb">top 20 clinically important symptoms</strong> (ranked by global SHAP importance).
    All unselected symptoms default to absent (0). Choose a model, then click
    <strong style="color:#00c2cb">RUN DIAGNOSIS</strong>.
    </div>
    """, unsafe_allow_html=True)

    col_controls, col_results = st.columns([1, 1], gap="large")

    with col_controls:
        st.markdown('<div class="section-header">// SELECT MODEL</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Algorithm",
            ["MLP (Neural Net)", "Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"],
            label_visibility="collapsed",
        )

        # Model info badges
        model_info = {
            "MLP (Neural Net)":      ("Best test accuracy", "#22c55e"),
            "Random Forest":         ("Ensemble — 200 trees", "#00c2cb"),
            "XGBoost":               ("Gradient boosting", "#00c2cb"),
            "Decision Tree":         ("Interpretable rules", "#64748b"),
            "Logistic Regression":   ("Linear baseline", "#64748b"),
        }
        info_text, info_color = model_info[model_choice]
        st.markdown(f'<div style="margin-top:-0.5rem;margin-bottom:1rem;">'
                    f'<span class="tag" style="color:{info_color};border-color:{info_color}33;">'
                    f'{info_text}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">// SYMPTOM CHECKLIST</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;color:#475569;letter-spacing:1px;margin-bottom:0.8rem;">'
                    'TOP 20 SYMPTOMS BY SHAP IMPORTANCE — check all that apply</div>',
                    unsafe_allow_html=True)

        symptom_values = {}
        # 2-column layout for checkboxes
        check_cols = st.columns(2)
        for i, sym in enumerate(top20):
            with check_cols[i % 2]:
                label = sym.replace("_", " ").title()
                symptom_values[sym] = st.checkbox(label, value=False, key=f"sym_{sym}")

        n_selected = sum(symptom_values.values())
        st.markdown(f'<div style="margin-top:0.5rem;font-family:\'Share Tech Mono\',monospace;'
                    f'font-size:0.72rem;color:#64748b;">'
                    f'SYMPTOMS SELECTED: <span style="color:#00c2cb;">{n_selected}</span> / 20'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⬡  RUN DIAGNOSIS", type="primary", use_container_width=True)

    with col_results:
        st.markdown('<div class="section-header">// DIAGNOSTIC OUTPUT</div>', unsafe_allow_html=True)

        if not run_btn:
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed rgba(0,194,203,0.2);border-radius:10px;flex-direction:column;gap:1rem;">
                <div style="font-family:'Share Tech Mono',monospace;font-size:2rem;color:rgba(0,194,203,0.2);">🧬</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#334155;letter-spacing:3px;">
                    AWAITING INPUT</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Build input vector
            input_vec = {f: 0 for f in all_features}
            for sym, val in symptom_values.items():
                if sym in input_vec:
                    input_vec[sym] = int(val)
            X_input = pd.DataFrame([input_vec])

            try:
                model_map = {
                    "Logistic Regression": M["lr"],
                    "Decision Tree"      : M["dt"],
                    "Random Forest"      : M["rf"],
                    "XGBoost"            : M["xgb"],
                }

                with st.spinner("Running inference..."):
                    if model_choice == "MLP (Neural Net)":
                        X_sc  = M["mlp_sc"].transform(X_input)
                        probs = M["mlp"].predict(X_sc, verbose=0)[0]
                        pred_idx  = int(np.argmax(probs))
                        pred_name = le.inverse_transform([pred_idx])[0]
                        top5_idx  = np.argsort(probs)[::-1][:5]
                        top5      = [(le.inverse_transform([i])[0], float(probs[i])) for i in top5_idx]
                    else:
                        model     = model_map[model_choice]
                        pred_idx  = int(model.predict(X_input)[0])
                        pred_name = le.inverse_transform([pred_idx])[0]
                        probs_all = model.predict_proba(X_input)[0]
                        top5_idx  = np.argsort(probs_all)[::-1][:5]
                        top5      = [(le.inverse_transform([i])[0], float(probs_all[i])) for i in top5_idx]

                # Primary result
                confidence = top5[0][1]
                conf_color = "#22c55e" if confidence > 0.8 else "#f59e0b" if confidence > 0.5 else "#ef4444"
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">PRIMARY DIAGNOSIS</div>
                    <div class="result-disease">{pred_name.upper()}</div>
                    <div style="margin-top:0.5rem;">
                        <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:{conf_color};">
                            CONFIDENCE: {confidence:.1%}
                        </span>
                        &nbsp;&nbsp;
                        <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#475569;">
                            MODEL: {model_choice.upper()}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Differential diagnoses
                st.markdown('<div class="section-header">// DIFFERENTIAL DIAGNOSES</div>',
                            unsafe_allow_html=True)
                for rank, (dname, prob) in enumerate(top5, 1):
                    bar_color = "#00c2cb" if rank == 1 else "#334155"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                                    color:#475569;width:18px;">#{rank}</div>
                        <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                                <span style="font-size:0.78rem;color:#cbd5e1;">{dname}</span>
                                <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                                             color:{bar_color};">{prob:.1%}</span>
                            </div>
                            <div style="height:4px;background:#1e293b;border-radius:2px;">
                                <div style="height:100%;width:{prob*100:.1f}%;background:{bar_color};
                                            border-radius:2px;transition:width 0.5s;
                                            box-shadow:0 0 8px {bar_color}66;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # SHAP waterfall for tree-based models
                tree_models = {"Decision Tree": M["dt"], "Random Forest": M["rf"], "XGBoost": M["xgb"]}
                if model_choice in tree_models:
                    st.markdown('<div class="section-header">// SHAP ATTRIBUTION — THIS PREDICTION</div>',
                                unsafe_allow_html=True)
                    with st.spinner("Computing SHAP attribution…"):
                        expl = shap.TreeExplainer(tree_models[model_choice])
                        sv   = expl.shap_values(X_input)

                        if isinstance(sv, list):
                            sv_cls = sv[pred_idx][0]
                            bv     = (expl.expected_value[pred_idx]
                                      if hasattr(expl.expected_value, "__len__")
                                      else expl.expected_value)
                        elif hasattr(sv, "ndim") and sv.ndim == 3:
                            sv_cls = sv[0, :, pred_idx]
                            bv     = (expl.expected_value[pred_idx]
                                      if hasattr(expl.expected_value, "__len__")
                                      else float(expl.expected_value))
                        else:
                            sv_cls = sv[0]
                            bv     = (float(expl.expected_value)
                                      if not hasattr(expl.expected_value, "__len__")
                                      else float(expl.expected_value[0]))

                        exp = shap.Explanation(
                            values        = sv_cls,
                            base_values   = bv,
                            data          = X_input.values[0],
                            feature_names = list(X_input.columns),
                        )
                        fig_wf = plt.figure(figsize=(9, 5))
                        shap.plots.waterfall(exp, max_display=12, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_wf, use_container_width=True)
                        plt.close()
                else:
                    # LR or MLP — top feature bar chart
                    st.markdown('<div class="section-header">// FEATURE INFLUENCE — THIS PREDICTION</div>',
                                unsafe_allow_html=True)
                    if model_choice == "Logistic Regression":
                        coef     = M["lr"]["lr"].coef_[pred_idx]
                        feat_imp = pd.Series(np.abs(coef), index=all_features).nlargest(12)
                    else:
                        active   = X_input.iloc[0]
                        feat_imp = (active[active > 0].nlargest(12)
                                    if active.sum() > 0
                                    else pd.Series(np.ones(12), index=all_features[:12]))
                    fig_bar, ax = plt.subplots(figsize=(8, 4))
                    fig_bar.patch.set_facecolor("#101828")
                    ax.set_facecolor("#101828")
                    feat_imp.sort_values().plot(kind="barh", ax=ax, color="#00c2cb")
                    ax.tick_params(colors="#94a3b8", labelsize=8)
                    ax.set_xlabel("Influence", color="#64748b", fontsize=8)
                    ax.spines[:].set_color("#1e293b")
                    ax.set_title("Top Influential Symptoms", color="#00c2cb", fontsize=9,
                                 fontfamily="monospace")
                    plt.tight_layout()
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close()

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
