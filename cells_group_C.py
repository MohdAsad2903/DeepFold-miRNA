# ═══════════════════════════════════════════════════════════════════════════════
# CELL GROUP C — RNA-FM Embeddings
# Copy each "# ═══ CELL ═══" section into a separate Jupyter notebook cell.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══ CELL C0 — Markdown ═══
# ---
# ## Step 12 — RNA-FM Language Model Embeddings
#
# RNA-FM is a foundation model pre-trained on 23M non-coding RNA sequences.
# Its 640-dim per-token embeddings encode evolutionary conservation, structural
# context, and family identity. We use FROZEN embeddings only — no fine-tuning
# at 2,372 samples. The difference embedding (healthy − mutant) captures how
# the SNP shifts the learned RNA representation.


# ═══ CELL C1 — Install and load RNA-FM ═══
import subprocess, sys

try:
    import fm
    print(f"RNA-FM already installed")
except ImportError:
    print("Installing RNA-FM (fair-esm)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fair-esm", "-q"],
                   check=True)
    import fm
    print("RNA-FM installed")

import torch
import numpy as np
import pandas as pd
import os

model_rnafm, alphabet = fm.pretrained.rna_fm_t12()
model_rnafm.eval()
if torch.cuda.is_available():
    model_rnafm = model_rnafm.cuda()
print("RNA-FM loaded (frozen, no fine-tuning)")

batch_converter = alphabet.get_batch_converter()


# ═══ CELL C2 — Extract frozen embeddings for all 2,372 samples ═══
df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_PATH = "DeepFold_Dataset/rnafm_embeddings.npy"

def get_rnafm_embedding(seq, model, batch_converter, device):
    """
    Extract frozen RNA-FM embedding (640 dims) via mean pooling over positions.
    RNA-FM uses DNA alphabet internally (T not U).
    """
    seq_dna = seq.upper().replace("U", "T")
    # Truncate to 1022 to avoid OOM (RNA-FM max is ~1024 with BOS/EOS)
    seq_dna = seq_dna[:1022]

    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq_dna)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    # Mean pool over positions, excluding BOS (idx 0) and EOS (idx -1)
    token_repr = results["representations"][12][0, 1:-1]  # (L, 640)
    return token_repr.mean(dim=0).cpu().numpy()   # (640,)


if os.path.exists(EMB_PATH):
    print(f"Loading cached embeddings from {EMB_PATH}")
    X_rnafm_diff = np.load(EMB_PATH)
    print(f"Embeddings shape: {X_rnafm_diff.shape}")
else:
    print(f"Extracting RNA-FM embeddings for {len(df)} samples...")
    print("(~5-10 min on GPU, ~30 min on CPU)")

    rnafm_diffs = []
    for idx, (_, row) in enumerate(df.iterrows()):
        emb_h = get_rnafm_embedding(row["Seq_Healthy"], model_rnafm,
                                     batch_converter, DEVICE)
        emb_m = get_rnafm_embedding(row["Seq_Mutant"], model_rnafm,
                                     batch_converter, DEVICE)
        rnafm_diffs.append(emb_h - emb_m)   # difference embedding

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(df)} done")

    X_rnafm_diff = np.stack(rnafm_diffs).astype(np.float32)
    np.save(EMB_PATH, X_rnafm_diff)
    print(f"Saved → {EMB_PATH}")

print(f"RNA-FM diff embeddings: {X_rnafm_diff.shape}")
print(f"Sample embedding range: [{X_rnafm_diff[0].min():.4f}, {X_rnafm_diff[0].max():.4f}]")

# Free GPU memory
del model_rnafm
torch.cuda.empty_cache()


# ═══ CELL C3 — RNA-FM + k-mer XGBoost classifier ═══
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# Combine k-mer v2 features + RNA-FM diff
X_combined = np.hstack([X_kmer_v2, X_rnafm_diff])
print(f"Combined feature matrix: {X_combined.shape}  "
      f"(k-mer_v2: {X_kmer_v2.shape[1]} + RNA-FM: {X_rnafm_diff.shape[1]})")

sgkf_rnafm = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

rnafm_fold_metrics = []
rnafm_all_probs    = []
rnafm_all_labels   = []
rnafm_fold_models  = []

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("DeepFold_models", exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(sgkf_rnafm.split(X_combined, y, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — RNA-FM + k-mer XGBoost")
    print(f"{'='*55}")

    X_tr, y_tr = X_combined[train_idx], y[train_idx]
    X_te, y_te = X_combined[test_idx],  y[test_idx]
    groups_tr  = groups[train_idx]

    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42+fold)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'eval_metric': 'logloss', 'use_label_encoder': False,
            'random_state': 42, 'verbosity': 0,
        }
        aucs = []
        for inner_tr, inner_val in inner_sgkf.split(X_tr, y_tr, groups_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[inner_tr], y_tr[inner_tr])
            p = clf.predict_proba(X_tr[inner_val])[:, 1]
            aucs.append(roc_auc_score(y_tr[inner_val], p))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    bp = study.best_params
    print(f"  Best inner AUC: {study.best_value:.4f}")

    model = XGBClassifier(
        n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],
        learning_rate=bp['lr'], subsample=bp['subsample'],
        colsample_bytree=bp['col'], min_child_weight=bp['mcw'],
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    rnafm_all_probs.extend(probs.tolist())
    rnafm_all_labels.extend(y_te.tolist())
    rnafm_fold_models.append(model)

    m = {
        "accuracy":  accuracy_score(y_te, preds),
        "auc":       roc_auc_score(y_te, probs),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall":    recall_score(y_te, preds, zero_division=0),
        "f1":        f1_score(y_te, preds, zero_division=0),
    }
    rnafm_fold_metrics.append(m)
    print(f"  Fold {fold+1} — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}")

    joblib.dump(model, f"DeepFold_models/rnafm_xgb_fold{fold}.pkl")

print(f"\n{'='*55}")
print("RNA-FM + k-mer XGBoost — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in rnafm_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nOOF AUC: {roc_auc_score(rnafm_all_labels, rnafm_all_probs):.4f}")
