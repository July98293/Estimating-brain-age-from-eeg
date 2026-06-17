

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(MODEL_DIR, "..", "out")
SPECTRAL_CSV  = os.path.join(OUT_DIR, "spectral_features.csv")
OUT_PRED_CSV  = os.path.join(OUT_DIR, "age_group_predictions.csv")
OUT_FEAT_PLOT = os.path.join(OUT_DIR, "age_model_feature_importance.png")
OUT_CM_PLOT   = os.path.join(OUT_DIR, "age_model_confusion_matrix.png")
OUT_ROC_PLOT  = os.path.join(OUT_DIR, "age_model_roc_curve.png")
OUT_MODEL     = os.path.join(OUT_DIR, "age_model.joblib")

AGE_THRESHOLD = 35
TOP_K         = 50
RANDOM_STATE  = 42

GBC_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
)

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(SPECTRAL_CSV)
df = df[df["session"] == "ses-t1"].reset_index(drop=True)

feature_cols = [c for c in df.columns
                if c not in ("participant_id", "session", "age", "sex")]

X        = df[feature_cols].values.astype(np.float64)
y        = (df["age"].values >= AGE_THRESHOLD).astype(int)
subjects = df["participant_id"].values
ages     = df["age"].values
n        = len(y)

print(f"Dataset: {n} subjects | young(<{AGE_THRESHOLD}): {(y==0).sum()} | "
      f"older(>={AGE_THRESHOLD}): {(y==1).sum()}")
print(f"Features: {X.shape[1]}  ->  SelectKBest keeps top {TOP_K}")
print(f"\nRunning Leave-One-Out CV ({n} folds) ...")

# ── LOO cross-validation ──────────────────────────────────────────────────────
loo      = LeaveOneOut()
oof_true = np.zeros(n, dtype=int)
oof_pred = np.zeros(n, dtype=int)
oof_prob = np.zeros(n)

for fold, (train_idx, test_idx) in enumerate(loo.split(X), 1):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=TOP_K)),
        ("gbc",    GradientBoostingClassifier(**GBC_PARAMS)),
    ])
    pipe.fit(X[train_idx], y[train_idx])
    oof_true[test_idx] = y[test_idx]
    oof_pred[test_idx] = pipe.predict(X[test_idx])
    oof_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    if fold % 10 == 0 or fold == n:
        print(f"  fold {fold}/{n}")

# ── metrics ───────────────────────────────────────────────────────────────────
acc     = accuracy_score(oof_true, oof_pred)
bal_acc = balanced_accuracy_score(oof_true, oof_pred)
auc     = roc_auc_score(oof_true, oof_prob)

print(f"\nLOO-CV results ({n} subjects, each tested exactly once):")
print(f"  Accuracy          : {acc:.3f}")
print(f"  Balanced accuracy : {bal_acc:.3f}")
print(f"  ROC-AUC           : {auc:.3f}")
print("\nClassification report (aggregated OOF predictions):")
print(classification_report(oof_true, oof_pred, target_names=["young", "older"]))

# ── save predictions ──────────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "participant_id": subjects,
    "age":            ages,
    "true_label":     oof_true,
    "true_group":     ["young" if v == 0 else "older" for v in oof_true],
    "pred_label":     oof_pred,
    "pred_group":     ["young" if v == 0 else "older" for v in oof_pred],
    "prob_older":     oof_prob.round(4),
    "correct":        oof_true == oof_pred,
})
pred_df.to_csv(OUT_PRED_CSV, index=False)
print(f"\nPredictions saved -> {OUT_PRED_CSV}")

# ── final model on ALL data ───────────────────────────────────────────────────
print("\nFitting final model on all subjects ...")
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(f_classif, k=TOP_K)),
    ("gbc",    GradientBoostingClassifier(**GBC_PARAMS)),
])
final_pipe.fit(X, y)
joblib.dump(final_pipe, OUT_MODEL)
print(f"Final model saved -> {OUT_MODEL}")

# ── feature importance ────────────────────────────────────────────────────────
selected_idx   = final_pipe.named_steps["select"].get_support(indices=True)
selected_names = [feature_cols[i] for i in selected_idx]
importances    = final_pipe.named_steps["gbc"].feature_importances_
order          = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(20), importances[order][::-1], color="steelblue")
ax.set_yticks(range(20))
ax.set_yticklabels([selected_names[i] for i in order][::-1], fontsize=8)
ax.set_xlabel("Feature importance (GBC, final model trained on all data)")
ax.set_title(f"Top 20 features | SelectKBest k={TOP_K} -> GBC")
plt.tight_layout()
plt.savefig(OUT_FEAT_PLOT, dpi=150)
plt.close()
print(f"Feature importance plot -> {OUT_FEAT_PLOT}")

# ── confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(oof_true, oof_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["young", "older"]).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion matrix (LOO-CV, n={n})\n"
             f"Acc={acc:.2f}  BalAcc={bal_acc:.2f}  AUC={auc:.2f}")
plt.tight_layout()
plt.savefig(OUT_CM_PLOT, dpi=150)
plt.close()
print(f"Confusion matrix plot -> {OUT_CM_PLOT}")

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(oof_true, oof_prob)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, lw=2, label=f"GBC (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve - LOO-CV | young vs older")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT_ROC_PLOT, dpi=150)
plt.close()
print(f"ROC curve plot -> {OUT_ROC_PLOT}")
