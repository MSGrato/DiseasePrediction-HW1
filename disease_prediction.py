"""
Disease Prediction — End-to-End Data Science Workflow
Runs as a plain Python script; every artifact is saved to disk so app.py can load them.
random_state=42 everywhere.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import seaborn as sns
import joblib
import shap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = "/Users/mirandagrato/Documents/DiseasePredict"
FIG    = f"{BASE}/figures"
MOD    = f"{BASE}/models"
os.makedirs(FIG, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 — Load & Verify
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 0: LOAD AND VERIFY DATA")
print("="*70)

train = pd.read_csv(f"{BASE}/Training.csv")
test  = pd.read_csv(f"{BASE}/Testing.csv")

# Drop the trailing empty column that comes from Training.csv
train = train.drop(columns=["Unnamed: 133"], errors="ignore")

# Strip accidental whitespace from target labels
train["prognosis"] = train["prognosis"].str.strip()
test["prognosis"]  = test["prognosis"].str.strip()

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"\nTrain dtypes (sample):\n{train.dtypes.value_counts()}")
print(f"\nTrain head:\n{train.head()}")
print(f"\nTrain describe (sample – first 5 feature cols):\n{train.iloc[:, :5].describe()}")
print(f"\nTarget value counts (train):\n{train['prognosis'].value_counts().to_string()}")
print(f"\nMissing values – train: {train.isnull().sum().sum()} | test: {test.isnull().sum().sum()}")
assert train.isnull().sum().sum() == 0, "Unexpected NaNs in training data!"
assert test.isnull().sum().sum()  == 0, "Unexpected NaNs in test data!"
print("✓ Zero missing values confirmed.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Descriptive Analytics
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: DESCRIPTIVE ANALYTICS")
print("="*70)

# ── 1.1 Dataset Introduction ─────────────────────────────────────────────────
print("""
DATASET INTRODUCTION
────────────────────
This is a disease prediction dataset. Each of the 4,920 rows represents a
unique patient encounter. The 132 feature columns are binary symptom
indicators (0 = symptom absent, 1 = symptom present). The target column
'prognosis' contains 41 possible disease diagnoses.

PREDICTION TASK:
Given a binary symptom vector, predict the most likely disease.

WHY IT MATTERS:
Automated disease pre-screening from symptom patterns could help
under-resourced clinics triage patients faster, reduce diagnostic delays,
and flag patients who need immediate specialist attention.

BASIC STATS:
  • Training rows  : 4,920  (120 rows × 41 classes — perfectly balanced)
  • Test rows      : 42
  • Feature columns: 132 (all binary 0/1)
  • Target classes : 41 diseases
  • Missing values : 0
""")

# ── 1.2 Target Distribution ───────────────────────────────────────────────────
vc = train["prognosis"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(10, 12))
vc.plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Number of patients")
ax.set_title("Disease Class Distribution (Training Set)")
plt.tight_layout()
fig.savefig(f"{FIG}/01_target_distribution.png", dpi=150)
plt.close()
print("✓ Saved 01_target_distribution.png")
print("""
Interpretation: Every disease class has exactly 120 training samples. This
perfect balance means accuracy is a valid primary metric and no resampling
(SMOTE, class weights, etc.) is required.
""")

# Feature matrix for analysis
features = train.drop(columns=["prognosis"])
n_rows   = len(features)

# ── 1.3 Feature Distributions & Relationships ────────────────────────────────

# --- Plot 1: Top-20 Most Common Symptoms ---
symptom_prevalence = features.sum() / n_rows
top20_symptoms = symptom_prevalence.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20_symptoms.sort_values().plot(kind="barh", ax=ax, color="coral")
ax.set_xlabel("Prevalence (fraction of patients)")
ax.set_title("Top 20 Most Common Symptoms")
plt.tight_layout()
fig.savefig(f"{FIG}/02_top20_symptoms.png", dpi=150)
plt.close()
print("✓ Saved 02_top20_symptoms.png")
print("""
Interpretation: The most prevalent symptoms (e.g. fatigue, high fever,
itching) appear in a large fraction of patients across many diseases. These
high-prevalence features have lower discriminative power on their own but
may form important interaction patterns with rarer symptoms.
""")

# --- Plot 2: Symptom Co-occurrence Heatmap (top 15) ---
top15 = symptom_prevalence.nlargest(15).index.tolist()
co_occurrence = features[top15].T.dot(features[top15]) / n_rows

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(co_occurrence, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Symptom Co-occurrence Rates (Top 15 Symptoms)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig(f"{FIG}/03_cooccurrence_heatmap.png", dpi=150)
plt.close()
print("✓ Saved 03_cooccurrence_heatmap.png")
print("""
Interpretation: High co-occurrence values (>0.3) on the diagonal-adjacent
cells indicate symptom pairs that frequently appear together, suggesting
shared underlying disease mechanisms. Blocks of high co-occurrence reveal
symptom clusters that likely correspond to specific disease families.
""")

# --- Plot 3: Symptoms per Patient Distribution ---
symptoms_per_patient = features.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(symptoms_per_patient, bins=30, color="mediumseagreen", edgecolor="white")
ax.set_xlabel("Number of Symptoms per Patient")
ax.set_ylabel("Number of Patients")
ax.set_title("Distribution of Symptom Count per Patient")
plt.tight_layout()
fig.savefig(f"{FIG}/04_symptoms_per_patient.png", dpi=150)
plt.close()
print("✓ Saved 04_symptoms_per_patient.png")
print(f"""
Interpretation: Patients have a mean of {symptoms_per_patient.mean():.1f} symptoms
(median {symptoms_per_patient.median():.0f}). The distribution is roughly unimodal,
suggesting disease presentations are similarly complex across classes. Models
need to handle variable symptom combinations rather than simple threshold rules.
""")

# --- Plot 4: Disease-Symptom Profile Heatmap (10 diseases × 20 symptoms) ---
sample_diseases = train["prognosis"].value_counts().index[:10].tolist()
top20_list = top20_symptoms.index.tolist()

disease_profiles = (
    train[train["prognosis"].isin(sample_diseases)]
    .groupby("prognosis")[top20_list]
    .mean()
)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(disease_profiles, cmap="Blues", ax=ax,
            linewidths=0.4, annot=True, fmt=".2f", annot_kws={"size": 7})
ax.set_title("Disease-Symptom Profile (10 Diseases × Top 20 Symptoms)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig(f"{FIG}/05_disease_symptom_profile.png", dpi=150)
plt.close()
print("✓ Saved 05_disease_symptom_profile.png")
print("""
Interpretation: Each row is a disease and each cell shows the average presence
of that symptom for patients with that disease. Distinct row patterns confirm
that each disease has a unique symptom signature, which is why tree-based
models can achieve near-perfect accuracy on this structured dataset.
""")

# ── 1.4 Correlation Heatmap ──────────────────────────────────────────────────
top30_var = features.var().nlargest(30).index.tolist()
corr_matrix = features[top30_var].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=ax,
            annot=False, linewidths=0.3, square=True)
ax.set_title("Correlation Heatmap — Top 30 Most Variable Symptoms", fontsize=12)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
fig.savefig(f"{FIG}/06_correlation_heatmap.png", dpi=150)
plt.close()
print("✓ Saved 06_correlation_heatmap.png")

# Find strongest off-diagonal correlation
corr_vals = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_corr = corr_vals.stack().abs().idxmax()
max_val  = corr_vals.stack().abs().max()
print(f"""
Correlation Insight: The strongest pairwise correlation is between
'{max_corr[0]}' and '{max_corr[1]}' (|r| = {max_val:.3f}).
High correlations suggest symptom clusters that tend to co-present within
specific disease families. These redundant features are handled gracefully
by tree-based models via their splitting mechanism, and LR handles them
through regularization. No manual collinearity pruning is needed.
""")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Data Preparation
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: DATA PREPARATION")
print("="*70)

# Separate features and target
X_train_full = train.drop(columns=["prognosis"])
y_train_full = train["prognosis"]

X_test = test.drop(columns=["prognosis"])
y_test = test["prognosis"]

# Encode string labels → integers (required by tree models and neural net)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_full)
y_test_enc  = le.transform(y_test)

# Save LabelEncoder so app.py can decode predictions back to disease names
joblib.dump(le, f"{MOD}/label_encoder.pkl")
print(f"✓ LabelEncoder saved  → {MOD}/label_encoder.pkl")
print(f"  Classes: {list(le.classes_[:5])} ... ({len(le.classes_)} total)")

# 70/30 split of training data for model development / hyperparameter tuning
# stratify=y_train_enc ensures each class keeps the same proportion in both splits
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_enc,
    test_size=0.3, random_state=42, stratify=y_train_enc
)
print(f"\nSplit sizes — X_train: {X_train.shape}, X_val: {X_val.shape}")

# No scaling needed for tree models (splits on threshold, order-invariant to scaling).
# Logistic Regression and MLP need StandardScaler → handled inside Pipeline / separate scaler.
# No missing value imputation (zero NaNs confirmed above).
# No feature encoding (all 132 columns are already integer 0/1).
print("No feature scaling applied globally (all features are binary 0/1).")
print("Scaling applied inside pipelines for LR and MLP only.")

# ─── Helper: evaluate any fitted model on val + test ─────────────────────────
def evaluate(name, model, X_v, y_v, X_t, y_t, le):
    """Return dict of metrics on validation and test sets."""
    yv_pred = model.predict(X_v)
    yt_pred = model.predict(X_t)

    def metrics(y_true, y_pred):
        return dict(
            accuracy  = accuracy_score(y_true, y_pred),
            f1_macro  = f1_score(y_true, y_pred, average="macro"),
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall    = recall_score(y_true, y_pred, average="macro", zero_division=0),
        )

    val_m  = metrics(y_v, yv_pred)
    test_m = metrics(y_t, yt_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Val  — Acc: {val_m['accuracy']:.4f}  F1: {val_m['f1_macro']:.4f}  "
          f"P: {val_m['precision']:.4f}  R: {val_m['recall']:.4f}")
    print(f"  Test — Acc: {test_m['accuracy']:.4f}  F1: {test_m['f1_macro']:.4f}  "
          f"P: {test_m['precision']:.4f}  R: {test_m['recall']:.4f}")

    # Confusion matrix on validation set
    cm  = confusion_matrix(y_v, yv_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name} (Validation Set)")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(f"{FIG}/cm_{safe_name}.png", dpi=120)
    plt.close()
    print(f"  ✓ Confusion matrix saved → figures/cm_{safe_name}.png")

    return test_m

results = {}   # will accumulate {model_name: metrics_dict}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Predictive Modeling
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: PREDICTIVE MODELING")
print("="*70)

# ── 3.1 Baseline — Logistic Regression ───────────────────────────────────────
print("\n[3.1] Logistic Regression (Pipeline: StandardScaler → LR)")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),               # LR sensitive to feature scale
    ("lr", LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",    # lbfgs handles multi-class natively (multinomial removed in sklearn 1.8)
        C=1.0              # default L2 regularization
    ))
])
lr_pipe.fit(X_train, y_train)
results["Logistic Regression"] = evaluate(
    "Logistic Regression", lr_pipe, X_val, y_val, X_test, y_test_enc, le
)
joblib.dump(lr_pipe, f"{MOD}/logistic_regression.pkl")
print(f"  ✓ Model saved → {MOD}/logistic_regression.pkl")

# ── 3.2 Decision Tree / CART ──────────────────────────────────────────────────
print("\n[3.2] Decision Tree (GridSearchCV 5-fold)")

dt_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={
        "max_depth"        : [5, 10, 15, None],
        "min_samples_leaf" : [1, 5, 10],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
dt_cv.fit(X_train, y_train)
best_dt = dt_cv.best_estimator_
print(f"  Best params: {dt_cv.best_params_}")
print(f"  Best CV F1 macro: {dt_cv.best_score_:.4f}")

results["Decision Tree"] = evaluate(
    "Decision Tree", best_dt, X_val, y_val, X_test, y_test_enc, le
)

# Visualise tree only if max_depth <= 5 (otherwise unreadable)
best_depth = dt_cv.best_params_["max_depth"]
if best_depth is not None and best_depth <= 5:
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_dt, feature_names=X_train.columns.tolist(),
              class_names=le.classes_, max_depth=3,
              filled=True, ax=ax, fontsize=7)
    ax.set_title("Decision Tree (depth ≤ 5)")
    plt.tight_layout()
    fig.savefig(f"{FIG}/decision_tree_plot.png", dpi=100)
    plt.close()
    print("  ✓ Tree plot saved → figures/decision_tree_plot.png")
else:
    print(f"  Tree depth={best_depth} — skipping visualisation (would be unreadable).")

joblib.dump(best_dt, f"{MOD}/decision_tree.pkl")
print(f"  ✓ Model saved → {MOD}/decision_tree.pkl")

# ── 3.3 Random Forest ─────────────────────────────────────────────────────────
print("\n[3.3] Random Forest (GridSearchCV 5-fold)")

rf_cv = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth"   : [10, 20, None],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_
print(f"  Best params: {rf_cv.best_params_}")
print(f"  Best CV F1 macro: {rf_cv.best_score_:.4f}")

results["Random Forest"] = evaluate(
    "Random Forest", best_rf, X_val, y_val, X_test, y_test_enc, le
)

# Feature importances — top 20 by mean decrease in impurity
fi = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top20_fi = fi.nlargest(20)
fig, ax = plt.subplots(figsize=(10, 7))
top20_fi.sort_values().plot(kind="barh", ax=ax, color="teal")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Random Forest — Top 20 Feature Importances")
plt.tight_layout()
fig.savefig(f"{FIG}/rf_feature_importances.png", dpi=150)
plt.close()
print("  ✓ Feature importances saved → figures/rf_feature_importances.png")

joblib.dump(best_rf, f"{MOD}/random_forest.pkl")
print(f"  ✓ Model saved → {MOD}/random_forest.pkl")

# ── 3.4 XGBoost ───────────────────────────────────────────────────────────────
print("\n[3.4] XGBoost (GridSearchCV 5-fold)")

xgb_cv = GridSearchCV(
    xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=41,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
    ),
    param_grid={
        "n_estimators" : [50, 100, 200],
        "max_depth"    : [3, 5, 6],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
xgb_cv.fit(X_train, y_train)
best_xgb = xgb_cv.best_estimator_
print(f"  Best params: {xgb_cv.best_params_}")
print(f"  Best CV F1 macro: {xgb_cv.best_score_:.4f}")

results["XGBoost"] = evaluate(
    "XGBoost", best_xgb, X_val, y_val, X_test, y_test_enc, le
)
joblib.dump(best_xgb, f"{MOD}/xgboost.pkl")
print(f"  ✓ Model saved → {MOD}/xgboost.pkl")

# ── 3.5 Neural Network — MLP (Keras) ─────────────────────────────────────────
print("\n[3.5] MLP Neural Network (Keras)")

# Scale features (neural nets are sensitive to input magnitude)
mlp_scaler = StandardScaler()
X_train_sc = mlp_scaler.fit_transform(X_train)
X_val_sc   = mlp_scaler.transform(X_val)
X_test_sc  = mlp_scaler.transform(X_test)
joblib.dump(mlp_scaler, f"{MOD}/mlp_scaler.pkl")
print(f"  ✓ MLP scaler saved → {MOD}/mlp_scaler.pkl")

# One-hot encode targets for categorical_crossentropy
y_train_ohe = to_categorical(y_train, num_classes=41)
y_val_ohe   = to_categorical(y_val,   num_classes=41)

# Build model
tf.random.set_seed(42)
mlp = keras.Sequential([
    layers.Input(shape=(132,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3, seed=42),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3, seed=42),
    layers.Dense(41, activation="softmax"),
])
mlp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
mlp.summary()

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = mlp.fit(
    X_train_sc, y_train_ohe,
    validation_data=(X_val_sc, y_val_ohe),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=0,
)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("MLP — Loss Curves")
axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(history.history["accuracy"],     label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("MLP — Accuracy Curves")
axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout()
fig.savefig(f"{FIG}/mlp_training_history.png", dpi=150)
plt.close()
print("  ✓ MLP training history saved → figures/mlp_training_history.png")

# Predict — argmax of softmax probabilities → integer class label
y_val_pred_mlp  = np.argmax(mlp.predict(X_val_sc,  verbose=0), axis=1)
y_test_pred_mlp = np.argmax(mlp.predict(X_test_sc, verbose=0), axis=1)

def metrics_dict(y_true, y_pred):
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        f1_macro  = f1_score(y_true, y_pred, average="macro"),
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0),
    )

mlp_val_m  = metrics_dict(y_val,      y_val_pred_mlp)
mlp_test_m = metrics_dict(y_test_enc, y_test_pred_mlp)
results["MLP (Neural Net)"] = mlp_test_m

print(f"\n  MLP Val  — Acc: {mlp_val_m['accuracy']:.4f}  F1: {mlp_val_m['f1_macro']:.4f}  "
      f"P: {mlp_val_m['precision']:.4f}  R: {mlp_val_m['recall']:.4f}")
print(f"  MLP Test — Acc: {mlp_test_m['accuracy']:.4f}  F1: {mlp_test_m['f1_macro']:.4f}  "
      f"P: {mlp_test_m['precision']:.4f}  R: {mlp_test_m['recall']:.4f}")

# MLP confusion matrix
cm_mlp = confusion_matrix(y_val, y_val_pred_mlp)
fig, ax = plt.subplots(figsize=(18, 16))
ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=le.classes_).plot(
    ax=ax, xticks_rotation=90, colorbar=False)
ax.set_title("Confusion Matrix — MLP Neural Net (Validation Set)")
plt.tight_layout()
fig.savefig(f"{FIG}/cm_mlp__neural_net_.png", dpi=120)
plt.close()
print("  ✓ MLP confusion matrix saved")

mlp.save(f"{MOD}/mlp_model.keras")
print(f"  ✓ MLP model saved → {MOD}/mlp_model.keras")

# ── 3.6 Model Comparison Summary ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3.6: MODEL COMPARISON SUMMARY (on held-out Test Set)")
print("="*70)

comparison_df = pd.DataFrame(results).T.reset_index()
comparison_df.columns = ["Model", "Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
comparison_df = comparison_df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(f"{BASE}/model_comparison.csv", index=False)

# Grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.2
metrics_to_plot = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
colors = ["steelblue", "coral", "mediumseagreen", "orchid"]
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    ax.bar(x + i * width, comparison_df[metric], width, label=metric, color=color)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison (Test Set)")
ax.legend()
plt.tight_layout()
fig.savefig(f"{FIG}/model_comparison.png", dpi=150)
plt.close()
print("✓ Saved figures/model_comparison.png")

best_model_name = comparison_df.iloc[0]["Model"]
best_f1         = comparison_df.iloc[0]["Macro F1"]
print(f"""
MODEL COMPARISON ANALYSIS:
The best-performing model on the held-out test set is {best_model_name}
with a macro F1 of {best_f1:.4f}. Tree-based ensemble models (Random Forest,
XGBoost) typically excel on this dataset because the binary symptom structure
creates clean decision boundaries. Logistic Regression provides a strong
interpretable baseline but may struggle with non-linear symptom interactions.
The MLP can capture complex patterns but requires more data to outperform
well-tuned tree ensembles on small structured datasets.
""")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — SHAP Explainability
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: SHAP EXPLAINABILITY")
print("="*70)

# Choose the best tree-based model for SHAP (RF or XGB, whichever has higher F1)
rf_f1  = results["Random Forest"]["f1_macro"]
xgb_f1 = results["XGBoost"]["f1_macro"]

if xgb_f1 >= rf_f1:
    shap_model      = best_xgb
    shap_model_name = "XGBoost"
    shap_X_val      = X_val
else:
    shap_model      = best_rf
    shap_model_name = "Random Forest"
    shap_X_val      = X_val

print(f"Using {shap_model_name} for SHAP (RF F1={rf_f1:.4f}, XGB F1={xgb_f1:.4f})")

# SHAP TreeExplainer — efficient for tree-based models
explainer   = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(shap_X_val)
# shap_values shape: (n_samples, n_features, n_classes) for RF
# or (n_samples, n_features) per class for XGB depending on version

print(f"  shap_values type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"  Number of class arrays: {len(shap_values)}, each shape: {shap_values[0].shape}")
else:
    print(f"  shap_values shape: {shap_values.shape}")

# --- SHAP Bar Plot: mean absolute SHAP across all classes ---
# Compute aggregated importance manually (mean |SHAP| over all samples & classes)
# to avoid shap.summary_plot rendering one bar per class (41 colored bars per feature).
if isinstance(shap_values, list):
    global_mean_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1))  # (n_features,)
elif shap_values.ndim == 3:
    global_mean_abs = np.abs(shap_values).mean(axis=(0, 2))            # (n_features,)
else:
    global_mean_abs = np.abs(shap_values).mean(axis=0)                 # (n_features,)

feat_names   = list(shap_X_val.columns)
importance_s = pd.Series(global_mean_abs, index=feat_names).nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#0a0f1e")
ax.set_facecolor("#0a0f1e")
bars = ax.barh(importance_s.index, importance_s.values, color="#00c2cb", height=0.65)
# Subtle gradient: fade bars from bright to dim
for i, bar in enumerate(bars):
    alpha = 0.5 + 0.5 * (i / len(bars))
    bar.set_color("#00c2cb")
    bar.set_alpha(alpha)
ax.set_xlabel("mean(|SHAP value|)  —  average impact on model output", color="#64748b", fontsize=9)
ax.set_title(f"SHAP — Mean Absolute Feature Importance ({shap_model_name})",
             color="#00c2cb", fontsize=11, fontfamily="monospace", pad=12)
ax.tick_params(axis="both", colors="#94a3b8", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#1e293b")
ax.spines["bottom"].set_color("#1e293b")
ax.xaxis.grid(True, color="#1e293b", linewidth=0.8, linestyle="--")
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(f"{FIG}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/shap_bar.png")

# --- SHAP Beeswarm summary plot ---
# For multi-class, use class 0 shap values as representative or collapse
if isinstance(shap_values, list):
    # list: [n_classes][n_samples, n_features]
    shap_arr        = np.array(shap_values)   # (n_classes, n_samples, n_features)
    shap_mean_abs   = np.abs(shap_arr).mean(axis=0)  # (n_samples, n_features)
    sv_for_beeswarm = shap_arr[0]             # show class-0 SHAP for beeswarm
elif shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) — XGBoost 3-D
    shap_mean_abs   = np.abs(shap_values).mean(axis=2)   # (n_samples, n_features)
    sv_for_beeswarm = shap_values[:, :, 0]               # class-0 slice
else:
    shap_mean_abs   = np.abs(shap_values)
    sv_for_beeswarm = shap_values

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv_for_beeswarm, shap_X_val,
                  plot_type="dot", show=False, max_display=20)
plt.title(f"SHAP Beeswarm Plot — Class 0 ({shap_model_name})")
plt.tight_layout()
fig.savefig(f"{FIG}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/shap_beeswarm.png")

# --- Waterfall plot for first sample ---
sample_idx       = 0
sample           = shap_X_val.iloc[[sample_idx]]
pred_class_idx   = shap_model.predict(sample)[0]
pred_class_name  = le.inverse_transform([int(pred_class_idx)])[0]
print(f"\n  Waterfall — Sample 0 predicted disease: '{pred_class_name}'")

# Build a shap.Explanation object for the predicted class.
# shap_values may be:
#   list of arrays      → shape [n_classes][n_samples, n_features]
#   3-D ndarray         → shape (n_samples, n_features, n_classes)
#   2-D ndarray         → shape (n_samples, n_features)   (single-output)
if isinstance(shap_values, list):
    # list indexing: shap_values[class][sample]
    sv_sample = shap_values[int(pred_class_idx)][sample_idx]
    base_val  = (explainer.expected_value[int(pred_class_idx)]
                 if hasattr(explainer.expected_value, "__len__")
                 else explainer.expected_value)
elif shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) — XGBoost multi-class ndarray
    sv_sample = shap_values[sample_idx, :, int(pred_class_idx)]
    base_val  = (explainer.expected_value[int(pred_class_idx)]
                 if hasattr(explainer.expected_value, "__len__")
                 else explainer.expected_value)
else:
    # 2-D single-output
    sv_sample = shap_values[sample_idx]
    base_val  = (float(explainer.expected_value)
                 if not hasattr(explainer.expected_value, "__len__")
                 else float(explainer.expected_value[0]))

exp = shap.Explanation(
    values        = sv_sample,
    base_values   = base_val,
    data          = sample.values[0],
    feature_names = list(shap_X_val.columns),
)

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.waterfall(exp, max_display=15, show=False)
plt.title(f"SHAP Waterfall — Sample 0 (Predicted: {pred_class_name})")
plt.tight_layout()
fig.savefig(f"{FIG}/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/shap_waterfall.png")

# Top contributing symptoms
top_symptoms_idx  = np.argsort(np.abs(sv_sample))[::-1][:5]
top_symptoms_list = [(shap_X_val.columns[i], sv_sample[i]) for i in top_symptoms_idx]
print("  Top 5 contributing symptoms for this prediction:")
for sym, val in top_symptoms_list:
    direction = "↑ pushes toward" if val > 0 else "↓ pushes away from"
    print(f"    {sym:40s}  SHAP={val:+.4f}  {direction} '{pred_class_name}'")

# Compute top 20 SHAP features globally for app widgets
if isinstance(shap_values, list):
    # list: [n_classes][n_samples, n_features] → mean over classes and samples
    global_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
elif shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) → mean over samples and classes
    global_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    global_shap = np.abs(shap_values).mean(axis=0)

top20_shap_features = list(
    pd.Series(global_shap, index=shap_X_val.columns).nlargest(20).index
)
joblib.dump(top20_shap_features, f"{MOD}/top20_shap_features.pkl")
print(f"\n  Top 20 SHAP features: {top20_shap_features[:5]} ...")
print(f"  ✓ Saved top20_shap_features.pkl")

print(f"""
SHAP GLOBAL INTERPRETATION:
The most globally important symptoms are: {', '.join(top20_shap_features[:5])}.
These features consistently shift model predictions regardless of disease class.
Positive SHAP values indicate the symptom increases the probability of the
predicted disease; negative values indicate it decreases it.
A clinician could use the waterfall plots to understand why the model predicted
a specific disease for a patient and to verify that the reasoning aligns with
clinical knowledge.
""")

# Save SHAP model name for app.py to know which model was used
joblib.dump(shap_model_name, f"{MOD}/shap_model_name.pkl")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save best model params for app.py
# ═════════════════════════════════════════════════════════════════════════════
best_params = {
    "Decision Tree": dt_cv.best_params_,
    "Random Forest": rf_cv.best_params_,
    "XGBoost"      : xgb_cv.best_params_,
}
joblib.dump(best_params, f"{MOD}/best_hyperparams.pkl")
print(f"\n✓ Best hyperparams saved → {MOD}/best_hyperparams.pkl")

# ─── Final Checklist ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DELIVERABLES CHECKLIST")
print("="*70)

checks = [
    ("figures/01_target_distribution.png",      f"{FIG}/01_target_distribution.png"),
    ("figures/02_top20_symptoms.png",            f"{FIG}/02_top20_symptoms.png"),
    ("figures/03_cooccurrence_heatmap.png",      f"{FIG}/03_cooccurrence_heatmap.png"),
    ("figures/04_symptoms_per_patient.png",      f"{FIG}/04_symptoms_per_patient.png"),
    ("figures/05_disease_symptom_profile.png",   f"{FIG}/05_disease_symptom_profile.png"),
    ("figures/06_correlation_heatmap.png",       f"{FIG}/06_correlation_heatmap.png"),
    ("figures/rf_feature_importances.png",       f"{FIG}/rf_feature_importances.png"),
    ("figures/mlp_training_history.png",         f"{FIG}/mlp_training_history.png"),
    ("figures/model_comparison.png",             f"{FIG}/model_comparison.png"),
    ("figures/shap_bar.png",                     f"{FIG}/shap_bar.png"),
    ("figures/shap_beeswarm.png",                f"{FIG}/shap_beeswarm.png"),
    ("figures/shap_waterfall.png",               f"{FIG}/shap_waterfall.png"),
    ("models/label_encoder.pkl",                 f"{MOD}/label_encoder.pkl"),
    ("models/logistic_regression.pkl",           f"{MOD}/logistic_regression.pkl"),
    ("models/decision_tree.pkl",                 f"{MOD}/decision_tree.pkl"),
    ("models/random_forest.pkl",                 f"{MOD}/random_forest.pkl"),
    ("models/xgboost.pkl",                       f"{MOD}/xgboost.pkl"),
    ("models/mlp_model.keras",                   f"{MOD}/mlp_model.keras"),
    ("models/mlp_scaler.pkl",                    f"{MOD}/mlp_scaler.pkl"),
    ("models/top20_shap_features.pkl",           f"{MOD}/top20_shap_features.pkl"),
    ("models/best_hyperparams.pkl",              f"{MOD}/best_hyperparams.pkl"),
    ("model_comparison.csv",                     f"{BASE}/model_comparison.csv"),
]

all_ok = True
for label, path in checks:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {label}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("ALL DELIVERABLES CREATED SUCCESSFULLY.")
else:
    print("Some files are missing — check output above.")
