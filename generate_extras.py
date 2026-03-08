"""
generate_extras.py
Generates three new artifacts and saves them to figures/ and models/:
  1. PyTorch MLP — proper EarlyStopping + epoch training history
  2. ROC curves  — one-vs-rest macro-averaged for all 5 models (validation set)
  3. Violin plot — symptom count distribution by disease class
Run after disease_prediction.py (requires saved models).
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, auc)

BASE = "/Users/mirandagrato/Documents/DiseasePredict"
FIG  = f"{BASE}/figures"
MOD  = f"{BASE}/models"

# ─── Load data & saved artifacts ─────────────────────────────────────────────
le    = joblib.load(f"{MOD}/label_encoder.pkl")
train = pd.read_csv(f"{BASE}/Training.csv")
test  = pd.read_csv(f"{BASE}/Testing.csv")
train = train.drop(columns=["Unnamed: 133"], errors="ignore")
train["prognosis"] = train["prognosis"].str.strip()
test["prognosis"]  = test["prognosis"].str.strip()

X_full     = train.drop(columns=["prognosis"])
y_enc      = le.transform(train["prognosis"])
X_test_raw = test.drop(columns=["prognosis"])
y_test_enc = le.transform(test["prognosis"])

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# Scaler (reuse the one saved by disease_prediction.py)
scaler      = joblib.load(f"{MOD}/mlp_scaler.pkl")
X_train_sc  = scaler.transform(X_train).astype(np.float32)
X_val_sc    = scaler.transform(X_val).astype(np.float32)
X_test_sc   = scaler.transform(X_test_raw).astype(np.float32)

N_CLASSES = len(le.classes_)   # 41
N_FEAT    = X_train_sc.shape[1]  # 132

# ═════════════════════════════════════════════════════════════════════════════
# 1. PYTORCH MLP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. PYTORCH MLP")
print("="*60)

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=128, output_dim=41):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(N_FEAT, 128, N_CLASSES)

# Convert to tensors
train_ds = TensorDataset(torch.from_numpy(X_train_sc),
                         torch.from_numpy(y_train.astype(np.int64)))
val_ds   = TensorDataset(torch.from_numpy(X_val_sc),
                         torch.from_numpy(y_val.astype(np.int64)))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# EarlyStopping state
patience    = 10
best_val    = float("inf")
no_improve  = 0
best_state  = None

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("Training (max 100 epochs, patience=10)...")
for epoch in range(1, 101):
    # ── Train ──
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * len(xb)
        t_correct += (logits.argmax(1) == yb).sum().item()
        t_total   += len(xb)

    # ── Validate ──
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            logits  = model(xb)
            loss    = criterion(logits, yb)
            v_loss    += loss.item() * len(xb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(xb)

    t_loss /= t_total
    v_loss /= v_total
    t_acc   = t_correct / t_total
    v_acc   = v_correct / v_total

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    # EarlyStopping
    if v_loss < best_val:
        best_val   = v_loss
        no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1

    if epoch % 10 == 0 or no_improve == patience:
        print(f"  Epoch {epoch:3d}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}"
              f"  train_acc={t_acc:.4f}  val_acc={v_acc:.4f}"
              + ("  ← best" if no_improve == 0 else f"  (no improve {no_improve}/{patience})"))

    if no_improve >= patience:
        print(f"  EarlyStopping triggered at epoch {epoch}. Restoring best weights.")
        model.load_state_dict(best_state)
        break

# ── Evaluate ──
model.eval()
with torch.no_grad():
    val_logits  = model(torch.from_numpy(X_val_sc))
    test_logits = model(torch.from_numpy(X_test_sc))

val_probs  = torch.softmax(val_logits,  dim=1).numpy()
test_probs = torch.softmax(test_logits, dim=1).numpy()
val_pred   = val_probs.argmax(axis=1)
test_pred  = test_probs.argmax(axis=1)

val_f1  = f1_score(y_val,      val_pred,  average="macro")
test_f1 = f1_score(y_test_enc, test_pred, average="macro")
print(f"\n  Val  F1={val_f1:.4f}  Acc={accuracy_score(y_val, val_pred):.4f}")
print(f"  Test F1={test_f1:.4f}  Acc={accuracy_score(y_test_enc, test_pred):.4f}")

# ── Save model ──
torch.save({
    "state_dict" : best_state,
    "input_dim"  : N_FEAT,
    "hidden_dim" : 128,
    "output_dim" : N_CLASSES,
}, f"{MOD}/mlp_pytorch.pt")
print(f"  ✓ Saved models/mlp_pytorch.pt")

# ── Training history figure ──
epochs_ran = len(history["train_loss"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0a0f1e")
for ax in axes:
    ax.set_facecolor("#101828")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_color("#1e293b")
    ax.xaxis.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")
    ax.yaxis.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

ep = range(1, epochs_ran + 1)
axes[0].plot(ep, history["train_loss"], color="#00c2cb",  label="Train", linewidth=1.5)
axes[0].plot(ep, history["val_loss"],   color="#f59e0b",  label="Val",   linewidth=1.5)
axes[0].set_title("MLP Loss (PyTorch)", color="#00c2cb", fontfamily="monospace", fontsize=10)
axes[0].set_xlabel("Epoch", color="#64748b", fontsize=8)
axes[0].legend(facecolor="#101828", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)

axes[1].plot(ep, history["train_acc"], color="#00c2cb", label="Train", linewidth=1.5)
axes[1].plot(ep, history["val_acc"],   color="#f59e0b", label="Val",   linewidth=1.5)
axes[1].set_title("MLP Accuracy (PyTorch)", color="#00c2cb", fontfamily="monospace", fontsize=10)
axes[1].set_xlabel("Epoch", color="#64748b", fontsize=8)
axes[1].legend(facecolor="#101828", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)

plt.tight_layout()
fig.savefig(f"{FIG}/mlp_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/mlp_training_history.png")

# ── Update model_comparison.csv ──
comp = pd.read_csv(f"{BASE}/model_comparison.csv")
comp.loc[comp["Model"] == "MLP (Neural Net)", "Accuracy"]        = accuracy_score(y_test_enc, test_pred)
comp.loc[comp["Model"] == "MLP (Neural Net)", "Macro F1"]        = f1_score(y_test_enc, test_pred, average="macro")
comp.loc[comp["Model"] == "MLP (Neural Net)", "Macro Precision"] = precision_score(y_test_enc, test_pred, average="macro", zero_division=0)
comp.loc[comp["Model"] == "MLP (Neural Net)", "Macro Recall"]    = recall_score(y_test_enc, test_pred, average="macro", zero_division=0)
comp = comp.sort_values("Macro F1", ascending=False).reset_index(drop=True)
comp.to_csv(f"{BASE}/model_comparison.csv", index=False)
print("  ✓ Updated model_comparison.csv")

# ═════════════════════════════════════════════════════════════════════════════
# 2. ROC CURVES — one-vs-rest macro-averaged (validation set, 1476 samples)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. ROC CURVES")
print("="*60)

# Binarize validation labels for one-vs-rest
y_val_bin = label_binarize(y_val, classes=range(N_CLASSES))  # (1476, 41)

def macro_roc(y_bin, y_prob):
    """Return (macro_fpr, macro_tpr, macro_auc) for a probability matrix."""
    fpr_list, tpr_list = [], []
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue
        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        fpr_list.append(fpr_c)
        tpr_list.append(tpr_c)
    all_fpr  = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_c, tpr_c in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(fpr_list)
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

# Collect probs from each model on X_val
lr_model  = joblib.load(f"{MOD}/logistic_regression.pkl")
dt_model  = joblib.load(f"{MOD}/decision_tree.pkl")
rf_model  = joblib.load(f"{MOD}/random_forest.pkl")
xgb_model = joblib.load(f"{MOD}/xgboost.pkl")

model_rocs = {
    "Logistic Regression": macro_roc(y_val_bin, lr_model.predict_proba(X_val)),
    "Decision Tree"      : macro_roc(y_val_bin, dt_model.predict_proba(X_val)),
    "Random Forest"      : macro_roc(y_val_bin, rf_model.predict_proba(X_val)),
    "XGBoost"            : macro_roc(y_val_bin, xgb_model.predict_proba(X_val)),
    "MLP (PyTorch)"      : macro_roc(y_val_bin, val_probs),
}

colors = ["#00c2cb", "#f59e0b", "#22c55e", "#a78bfa", "#f87171"]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0a0f1e")
ax.set_facecolor("#101828")

ax.plot([0, 1], [0, 1], color="#334155", linestyle="--", linewidth=1, label="Random (AUC=0.50)")
for (name, (fpr, tpr, roc_auc)), color in zip(model_rocs.items(), colors):
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name}  (AUC={roc_auc:.4f})")
    print(f"  {name:<25} AUC={roc_auc:.4f}")

ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=9)
ax.set_ylabel("True Positive Rate",  color="#94a3b8", fontsize=9)
ax.set_title("ROC Curves — Macro One-vs-Rest (Validation Set)",
             color="#00c2cb", fontfamily="monospace", fontsize=10, pad=12)
ax.tick_params(colors="#64748b", labelsize=8)
ax.spines[:].set_color("#1e293b")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.xaxis.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")
ax.yaxis.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")
ax.set_axisbelow(True)
legend = ax.legend(facecolor="#0d1f36", edgecolor="#1e293b",
                   labelcolor="#cbd5e1", fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(f"{FIG}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/roc_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# 3. VIOLIN PLOT — Symptom count distribution by disease class
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. VIOLIN PLOT")
print("="*60)

# Compute symptom count (row sum) for each patient in the training set
symptom_count = X_full.sum(axis=1).rename("symptom_count")
disease_labels = train["prognosis"]

violin_df = pd.DataFrame({
    "symptom_count": symptom_count.values,
    "disease":       disease_labels.values,
})

# Pick 15 diseases with highest variance in symptom count (most visually interesting)
disease_var = violin_df.groupby("disease")["symptom_count"].var().nlargest(15)
top15 = disease_var.index.tolist()
plot_df = violin_df[violin_df["disease"].isin(top15)].copy()

# Sort diseases by median symptom count for a clean visual ordering
order = (plot_df.groupby("disease")["symptom_count"]
         .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0a0f1e")
ax.set_facecolor("#101828")

sns.violinplot(
    data=plot_df, x="disease", y="symptom_count",
    order=order, ax=ax,
    palette=sns.color_palette("blend:#00c2cb,#0066cc", n_colors=15),
    inner="box",        # draw mini box plot inside each violin
    linewidth=0.8,
    cut=0,              # don't extend beyond data range
)

ax.set_title("Symptom Count Distribution by Disease Class (Top 15 by Variance)",
             color="#00c2cb", fontfamily="monospace", fontsize=10, pad=12)
ax.set_xlabel("")
ax.set_ylabel("Number of Symptoms per Patient", color="#94a3b8", fontsize=9)
ax.tick_params(axis="x", colors="#94a3b8", labelsize=7, rotation=35)
ax.tick_params(axis="y", colors="#94a3b8", labelsize=8)
ax.spines[:].set_color("#1e293b")
ax.yaxis.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")
ax.set_axisbelow(True)
plt.xticks(ha="right")
plt.tight_layout()
fig.savefig(f"{FIG}/violin_symptom_count.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved figures/violin_symptom_count.png")

# ─── Final checklist ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE")
print("="*60)
for label, path in [
    ("models/mlp_pytorch.pt",           f"{MOD}/mlp_pytorch.pt"),
    ("figures/mlp_training_history.png", f"{FIG}/mlp_training_history.png"),
    ("figures/roc_curves.png",           f"{FIG}/roc_curves.png"),
    ("figures/violin_symptom_count.png", f"{FIG}/violin_symptom_count.png"),
    ("model_comparison.csv",             f"{BASE}/model_comparison.csv"),
]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {status}  {label}")
