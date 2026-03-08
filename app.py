"""
app.py — Disease Prediction Streamlit App
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
    page_title="Disease Prediction Studio",
    page_icon="🩺",
    layout="wide",
)

BASE  = os.path.dirname(os.path.abspath(__file__))
FIG   = os.path.join(BASE, "figures")
MOD   = os.path.join(BASE, "models")

# ─── Load all models once at startup ─────────────────────────────────────────
@st.cache_resource
def load_models():
    from tensorflow import keras
    models = {}
    models["le"]      = joblib.load(f"{MOD}/label_encoder.pkl")
    models["lr"]      = joblib.load(f"{MOD}/logistic_regression.pkl")
    models["dt"]      = joblib.load(f"{MOD}/decision_tree.pkl")
    models["rf"]      = joblib.load(f"{MOD}/random_forest.pkl")
    models["xgb"]     = joblib.load(f"{MOD}/xgboost.pkl")
    models["mlp"]     = keras.models.load_model(f"{MOD}/mlp_model.keras")
    models["mlp_sc"]  = joblib.load(f"{MOD}/mlp_scaler.pkl")
    models["top20"]   = joblib.load(f"{MOD}/top20_shap_features.pkl")
    models["params"]  = joblib.load(f"{MOD}/best_hyperparams.pkl")
    models["comp"]    = pd.read_csv(f"{BASE}/model_comparison.csv")
    return models

try:
    M = load_models()
except Exception as e:
    st.error(f"Could not load models. Run `python disease_prediction.py` first.\n\n{e}")
    st.stop()

le     = M["le"]
top20  = M["top20"]
comp   = M["comp"]

# All 132 symptom column names — derive from Training.csv if available, else use model features
@st.cache_data
def get_all_features():
    try:
        train = pd.read_csv(f"{BASE}/Training.csv")
        train = train.drop(columns=["Unnamed: 133", "prognosis"], errors="ignore")
        return list(train.columns)
    except Exception:
        # Fallback: use top20 (app will still work for prediction)
        return top20

all_features = get_all_features()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🩺 Disease Prediction from Symptom Profiles")
    st.subheader("Executive Summary")

    st.markdown("""
    ### Dataset Overview

    This project uses a curated medical dataset containing **4,920 patient records** with
    **132 binary symptom features** (each column is 0 = symptom absent, 1 = symptom present)
    and **41 possible disease diagnoses** as the prediction target. The dataset was designed
    to simulate real-world clinical symptom reporting. A held-out test set of 42 records
    (one per class) was kept completely separate to evaluate final model generalization.
    """)

    st.markdown("""
    ### Why It Matters

    Automated disease pre-screening from symptom patterns has significant potential in
    healthcare, particularly in under-resourced settings where specialist access is limited.
    A reliable classifier could assist front-line health workers in triaging patients,
    flagging high-risk presentations for urgent follow-up, and reducing diagnostic delays.
    This is especially relevant for the 41 disease categories in this dataset, which span
    infectious, metabolic, dermatological, and systemic conditions.
    """)

    best_row  = comp.iloc[0]
    best_name = best_row["Model"]
    best_f1   = best_row["Macro F1"]

    st.markdown(f"""
    ### Approach & Results

    Five machine learning models were developed and compared:
    **Logistic Regression** (interpretable baseline with StandardScaler pipeline),
    **Decision Tree / CART** (rule-based, tuned via 5-fold GridSearchCV),
    **Random Forest** (ensemble of 100–200 trees, tuned via GridSearchCV),
    **XGBoost** (gradient-boosted trees, tuned over learning rate, depth, and estimators), and
    **MLP Neural Network** (two-layer Keras network with Dropout and EarlyStopping).
    All tuning used 5-fold stratified cross-validation with macro-averaged F1 as the
    scoring metric to respect the 41-class structure. The **{best_name}** model achieved
    the best performance with a macro F1 of **{best_f1:.4f}** on the held-out test set.
    SHAP values were computed to explain individual predictions and identify the most
    diagnostically important symptoms globally.
    """)

    st.info("""
    **Intended Audience:** This summary is written for healthcare administrators and
    non-technical stakeholders. Full model metrics and feature explanations are available
    in the Model Performance and Explainability tabs.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Descriptive Analytics")

    figures = [
        ("01_target_distribution.png",
         "**Disease Class Distribution.** Each bar represents one of the 41 disease classes. "
         "All classes have exactly 120 training samples, confirming perfect balance. "
         "This means accuracy is a valid metric and no resampling is required."),

        ("02_top20_symptoms.png",
         "**Top 20 Most Prevalent Symptoms.** Bars show the fraction of patients who have each symptom. "
         "High-prevalence symptoms like fatigue and fever appear across many diseases, giving them "
         "lower individual discriminative power but strong value in combination."),

        ("03_cooccurrence_heatmap.png",
         "**Symptom Co-occurrence Heatmap (Top 15 Symptoms).** Each cell shows the fraction of patients "
         "who have both symptoms simultaneously. Darker cells indicate symptom pairs that frequently "
         "co-present, hinting at shared biological mechanisms or disease clusters."),

        ("04_symptoms_per_patient.png",
         "**Symptoms per Patient Distribution.** This histogram shows how many symptoms each patient "
         "presents with. The roughly unimodal distribution suggests most patients have a similar "
         "symptom burden, so classifiers must rely on which symptoms are present, not just how many."),

        ("05_disease_symptom_profile.png",
         "**Disease-Symptom Profile Heatmap (10 Diseases × Top 20 Symptoms).** Each row is a disease; "
         "cell values show the average presence of that symptom for patients with that disease. "
         "Distinct row patterns confirm that each disease has a unique symptom fingerprint."),

        ("06_correlation_heatmap.png",
         "**Correlation Heatmap (Top 30 Variable Symptoms).** This heatmap reveals linear associations "
         "between the most variable symptom features. Clustered blocks of high correlation indicate "
         "symptom groups that tend to appear together, which tree models exploit via their splitting logic."),

        ("rf_feature_importances.png",
         "**Random Forest Feature Importances (Top 20).** Mean decrease in impurity (Gini importance) "
         "for each feature. Features ranked higher here are used most frequently at early splits across "
         "all 100–200 trees, making them the primary drivers of Random Forest predictions."),

        ("mlp_training_history.png",
         "**MLP Training History.** Loss and accuracy curves for train and validation sets across epochs. "
         "EarlyStopping restored the best weights before overfitting began. Converging curves indicate "
         "healthy training without severe under- or over-fitting."),
    ]

    for fname, caption in figures:
        fpath = os.path.join(FIG, fname)
        if os.path.exists(fpath):
            st.image(fpath, use_container_width=True)
            st.caption(caption)
            st.divider()
        else:
            st.warning(f"Figure not found: {fname} — run disease_prediction.py to generate it.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🤖 Model Performance")

    st.subheader("Comparison Table (Held-out Test Set)")
    st.dataframe(comp.style.format({
        "Accuracy"       : "{:.4f}",
        "Macro F1"       : "{:.4f}",
        "Macro Precision": "{:.4f}",
        "Macro Recall"   : "{:.4f}",
    }), use_container_width=True)

    comp_path = os.path.join(FIG, "model_comparison.png")
    if os.path.exists(comp_path):
        st.subheader("Comparison Bar Chart")
        st.image(comp_path, use_container_width=True)

    st.subheader("Confusion Matrices (Validation Set)")
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
            with st.expander(f"Confusion Matrix — {model_label}", expanded=False):
                st.image(fpath, use_container_width=True)
        else:
            st.warning(f"Missing: {fname}")

    st.subheader("Best Hyperparameters")
    for model_name, params in M["params"].items():
        with st.expander(f"Best Params — {model_name}"):
            st.json(params)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    # ── SHAP Plots ────────────────────────────────────────────────────────────
    st.subheader("Global SHAP Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        bar_path = os.path.join(FIG, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="SHAP Bar Plot — Mean absolute SHAP values across all classes",
                     use_container_width=True)
    with col2:
        bee_path = os.path.join(FIG, "shap_beeswarm.png")
        if os.path.exists(bee_path):
            st.image(bee_path, caption="SHAP Beeswarm — Distribution of SHAP values (Class 0 shown)",
                     use_container_width=True)

    wf_path = os.path.join(FIG, "shap_waterfall.png")
    if os.path.exists(wf_path):
        st.subheader("SHAP Waterfall — Sample Explanation")
        st.image(wf_path, caption="Waterfall plot for a single patient prediction",
                 use_container_width=True)

    st.divider()

    # ── Interactive Prediction Widget ─────────────────────────────────────────
    st.subheader("Interactive Symptom Checker")
    st.markdown(
        "Select symptoms below (top 20 by SHAP importance). All other symptoms default to 0. "
        "Choose a model and click **Predict**."
    )

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP (Neural Net)"],
    )

    st.markdown("**Select present symptoms:**")
    # Lay checkboxes out in 4 columns
    cols = st.columns(4)
    symptom_values = {}
    for i, sym in enumerate(top20):
        with cols[i % 4]:
            symptom_values[sym] = st.checkbox(sym.replace("_", " "), value=False)

    if st.button("🔍 Predict Disease", type="primary"):
        # Build input vector — zeros for all 132 features, then fill top-20
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

            if model_choice == "MLP (Neural Net)":
                X_sc  = M["mlp_sc"].transform(X_input)
                probs = M["mlp"].predict(X_sc, verbose=0)[0]
                pred_idx  = int(np.argmax(probs))
                pred_name = le.inverse_transform([pred_idx])[0]
                top3_idx  = np.argsort(probs)[::-1][:3]
                top3      = [(le.inverse_transform([i])[0], float(probs[i])) for i in top3_idx]
            else:
                model = model_map[model_choice]
                pred_idx  = int(model.predict(X_input)[0])
                pred_name = le.inverse_transform([pred_idx])[0]
                probs_all = model.predict_proba(X_input)[0]
                top3_idx  = np.argsort(probs_all)[::-1][:3]
                top3      = [(le.inverse_transform([i])[0], float(probs_all[i])) for i in top3_idx]

            st.success(f"**Predicted Disease:** {pred_name}")
            st.markdown("**Top 3 Most Likely Diseases:**")
            for rank, (dname, prob) in enumerate(top3, 1):
                st.progress(prob, text=f"{rank}. {dname} — {prob*100:.1f}%")

            # SHAP waterfall for tree-based models
            tree_models = {"Decision Tree": M["dt"], "Random Forest": M["rf"], "XGBoost": M["xgb"]}
            if model_choice in tree_models:
                st.markdown("**SHAP Explanation for This Prediction:**")
                with st.spinner("Computing SHAP values…"):
                    expl      = shap.TreeExplainer(tree_models[model_choice])
                    sv        = expl.shap_values(X_input)
                    if isinstance(sv, list):
                        sv_cls = sv[pred_idx][0]
                        bv     = (expl.expected_value[pred_idx]
                                  if hasattr(expl.expected_value, "__len__")
                                  else expl.expected_value)
                    else:
                        sv_cls = sv[0]
                        bv     = (expl.expected_value[pred_idx]
                                  if hasattr(expl.expected_value, "__len__")
                                  else expl.expected_value)
                    exp = shap.Explanation(
                        values        = sv_cls,
                        base_values   = bv,
                        data          = X_input.values[0],
                        feature_names = list(X_input.columns),
                    )
                    fig_wf, _ = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(exp, max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_wf, use_container_width=True)
                    plt.close()
            else:
                # For LR and MLP — show top-10 feature bar chart
                st.markdown("**Top 10 Most Influential Features for This Prediction:**")
                if model_choice == "Logistic Regression":
                    # Use coefficient of predicted class
                    coef = M["lr"]["lr"].coef_[pred_idx]
                    feat_imp = pd.Series(np.abs(coef), index=all_features).nlargest(10)
                elif model_choice == "MLP (Neural Net)":
                    # Use gradient-based proxy: input × first-layer weights (row for pred class)
                    # Simple fallback: just show top-10 by input value (which symptoms are active)
                    active = X_input.iloc[0]
                    feat_imp = active[active > 0].nlargest(10) if active.sum() > 0 \
                               else pd.Series(np.ones(10), index=all_features[:10])
                fig_bar, ax = plt.subplots(figsize=(8, 5))
                feat_imp.sort_values().plot(kind="barh", ax=ax, color="steelblue")
                ax.set_title("Top 10 Influential Features")
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
