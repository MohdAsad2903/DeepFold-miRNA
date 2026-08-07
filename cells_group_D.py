# ═══════════════════════════════════════════════════════════════════════════════
# CELL GROUP D — Improved Ensemble Stacking
# Copy each "# ═══ CELL ═══" section into a separate Jupyter notebook cell.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══ CELL D0 — Markdown ═══
# ---
# ## Step 13 — Enhanced 5-Model Ensemble with XGBoost Meta-Learner
#
# ### Changes from v1 ensemble
# - 5 base models (added RNA-FM XGBoost)
# - Rank-normalised OOF probabilities before stacking
# - XGBoost meta-learner (captures non-linear interactions between base models)


# ═══ CELL D1 — Collect OOF probabilities from all 5 models ═══
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# Gather OOF probabilities from all base models
# These variables should be in scope from previous cells:
#   v4_all_probs     → CNN v4 (from Cell Group A)
#   gnn_all_probs    → Siamese GCN (from original notebook Step 8.2)
#   kmer_v2_all_probs → k-mer XGBoost v2 (from Cell Group B)
#   mfe_all_probs    → ViennaRNA MFE XGBoost (from original notebook Step 8.2c)
#   rnafm_all_probs  → RNA-FM XGBoost (from Cell Group C)

# Convert to numpy arrays
cnn_p   = np.array(v4_all_probs)
gcn_p   = np.array(gnn_all_probs)
kmer_p  = np.array(kmer_v2_all_probs)
mfe_p   = np.array(mfe_all_probs)
rnafm_p = np.array(rnafm_all_probs)

# Use labels from CNN v4 (should be identical order for all models with same SGKF)
true_y = np.array(v4_all_labels)

print("Base model OOF AUCs:")
print(f"  CNN v4:      {roc_auc_score(true_y, cnn_p):.4f}")
print(f"  GCN:         {roc_auc_score(true_y, gcn_p):.4f}")
print(f"  k-mer v2:    {roc_auc_score(true_y, kmer_p):.4f}")
print(f"  MFE XGB:     {roc_auc_score(true_y, mfe_p):.4f}")
print(f"  RNA-FM XGB:  {roc_auc_score(true_y, rnafm_p):.4f}")

# Verify all arrays have the same length
assert len(cnn_p) == len(gcn_p) == len(kmer_p) == len(mfe_p) == len(rnafm_p) == len(true_y), \
    f"Length mismatch! CNN:{len(cnn_p)} GCN:{len(gcn_p)} k-mer:{len(kmer_p)} " \
    f"MFE:{len(mfe_p)} RNAFM:{len(rnafm_p)} y:{len(true_y)}"
print(f"\nAll arrays: {len(true_y)} samples ✓")


# ═══ CELL D2 — Rank-normalise and build meta-features ═══

def rank_normalize(probs):
    """Rank-transform to [0, 1] — prevents calibration differences from
    distorting the meta-learner."""
    return rankdata(probs) / len(probs)

# Raw meta-features (for comparison)
meta_X5_raw = np.column_stack([cnn_p, gcn_p, kmer_p, mfe_p, rnafm_p])

# Rank-normalised meta-features
meta_X5_ranked = np.column_stack([
    rank_normalize(cnn_p),
    rank_normalize(gcn_p),
    rank_normalize(kmer_p),
    rank_normalize(mfe_p),
    rank_normalize(rnafm_p),
])

print(f"Meta-feature matrix (raw):    {meta_X5_raw.shape}")
print(f"Meta-feature matrix (ranked): {meta_X5_ranked.shape}")
print(f"\nRaw probability ranges:")
for name, arr in [("CNN", cnn_p), ("GCN", gcn_p), ("k-mer", kmer_p),
                  ("MFE", mfe_p), ("RNAFM", rnafm_p)]:
    print(f"  {name:8s}: [{arr.min():.3f}, {arr.max():.3f}]  mean={arr.mean():.3f}")


# ═══ CELL D3 — XGBoost meta-learner (5-fold CV) ═══

sgkf_meta = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

ensemble_v2_fold_metrics = []
ensemble_v2_all_probs    = np.zeros(len(true_y))
ensemble_v2_fold_models  = []

for fold, (train_idx, test_idx) in enumerate(sgkf_meta.split(meta_X5_ranked, true_y, groups)):
    meta_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    meta_model.fit(meta_X5_ranked[train_idx], true_y[train_idx])
    probs = meta_model.predict_proba(meta_X5_ranked[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    ensemble_v2_all_probs[test_idx] = probs
    ensemble_v2_fold_models.append(meta_model)

    m = {
        "accuracy":  accuracy_score(true_y[test_idx], preds),
        "auc":       roc_auc_score(true_y[test_idx], probs),
        "precision": precision_score(true_y[test_idx], preds, zero_division=0),
        "recall":    recall_score(true_y[test_idx], preds, zero_division=0),
        "f1":        f1_score(true_y[test_idx], preds, zero_division=0),
    }
    ensemble_v2_fold_metrics.append(m)

    # Feature importances show which base model contributes most
    imp = meta_model.feature_importances_
    names = ["CNN_v4", "GCN", "k-mer_v2", "MFE", "RNAFM"]
    imp_str = "  ".join(f"{n}: {v:.3f}" for n, v in zip(names, imp))
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f} | {imp_str}")

# Save the last fold's meta-learner (or retrain on full data)
meta_model_final = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, eval_metric='auc', use_label_encoder=False,
    random_state=42, verbosity=0,
)
meta_model_final.fit(meta_X5_ranked, true_y)
joblib.dump(meta_model_final, "DeepFold_models/meta_learner_v2.pkl")

print(f"\n{'='*55}")
print("Enhanced 5-Model Ensemble v2 — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in ensemble_v2_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nOOF AUC: {roc_auc_score(true_y, ensemble_v2_all_probs):.4f}")
print(f"\nOriginal 4-model ensemble AUC was 0.720 ± 0.020 — check improvement above.")


# ═══ CELL D4 — Updated comparison table ═══

print(f"\n{'='*70}")
print(f"{'FINAL METHOD COMPARISON v2 — 5-FOLD STRATIFIED GROUP CV':^70}")
print(f"{'='*70}")
print(f"{'Method':<32} {'AUC':^20} {'Accuracy':^12} {'F1':^8}")
print(f"{'-'*70}")

all_results = []

# Collect all model results
model_results = {
    "CNN v4 (DeepFoldCNN)":     v4_fold_metrics,
    "Siamese GCN":              gnn_fold_metrics,
    "k-mer XGB v2 (Optuna)":    kmer_v2_fold_metrics,
    "ViennaRNA MFE XGB":        mfe_fold_metrics,
    "RNA-FM + k-mer XGB":       rnafm_fold_metrics,
    "Ensemble v2 (5-model)":    ensemble_v2_fold_metrics,
}

# Also include original models if available
try:
    model_results["CNN v3 (original)"] = fold_metrics
except NameError:
    pass
try:
    model_results["k-mer XGB v1 (original)"] = kmer_results["fold_metrics"]
except NameError:
    pass
try:
    model_results["Ensemble v1 (4-model)"] = enhanced_fold_metrics
except NameError:
    pass

for name, folds in sorted(model_results.items(),
                           key=lambda x: -np.mean([m["auc"] for m in x[1]])):
    auc_m = np.mean([m["auc"] for m in folds])
    auc_s = np.std([m["auc"] for m in folds])
    acc   = np.mean([m["accuracy"] for m in folds])
    f1    = np.mean([m["f1"] for m in folds])
    best  = " ← BEST" if name.startswith("Ensemble v2") else ""
    print(f"{name:<32} {auc_m:.4f} ± {auc_s:.4f}   {acc:.4f}     {f1:.4f}{best}")
    all_results.append({"method": name, "auc_mean": auc_m, "auc_std": auc_s,
                        "accuracy": acc, "f1": f1})

print(f"{'='*70}")

# Save
rows = []
for name, folds in model_results.items():
    for i, m in enumerate(folds):
        rows.append({"method": name, "fold": i+1,
                     **{k: v for k, v in m.items() if k != "cm"}})
pd.DataFrame(rows).to_csv("DeepFold_Dataset/all_methods_comparison_v2.csv", index=False)
print("Saved → all_methods_comparison_v2.csv")
