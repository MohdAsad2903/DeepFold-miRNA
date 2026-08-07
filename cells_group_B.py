# ═══════════════════════════════════════════════════════════════════════════════
# CELL GROUP B — Improved k-mer XGBoost
# Copy each "# ═══ CELL ═══" section into a separate Jupyter notebook cell.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══ CELL B0 — Markdown ═══
# ---
# ## Step 11 — Improved k-mer XGBoost (v2)
#
# ### Changes from v1
# - **5-mer difference features** (+1,024 dims): captures pentamer motifs encoding RNA structural signals
# - **Local context window** (±5 nt around SNP): 5 focused position-aware features
# - **Optuna hyperparameter tuning**: Bayesian optimisation (50 trials) with inner 3-fold CV
# - Uses `XGBClassifier` instead of `GradientBoostingClassifier`
# - Total feature vector: 971 + 1024 + 5 = 2,000 dims


# ═══ CELL B1 — Extended feature extraction ═══
from itertools import product as iproduct
from collections import Counter
import numpy as np
import pandas as pd
import math as _math

df     = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
groups = df["miRNA_ID"].values
y      = df["Label"].values

# ── 5-mer vocabulary ─────────────────────────────────────────────────────────
bases_4   = ['A', 'U', 'C', 'G']
kmers_5   = [''.join(k) for k in iproduct(bases_4, repeat=5)]   # 1024
kmer5_idx = {km: i for i, km in enumerate(kmers_5)}


def kmer_freq(seq, k):
    """Frequency vector for k-mers of length k."""
    bases  = "ACGU"
    kmers  = [''.join(p) for p in iproduct(bases, repeat=k)]
    km_idx = {km: i for i, km in enumerate(kmers)}
    seq    = seq.upper().replace("T", "U")
    counts = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in km_idx:
            counts[km_idx[kmer]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def get_5mer_diff(seq_h, seq_m):
    """5-mer DIFFERENCE vector only (1024 dims)."""
    seq_h = seq_h.upper().replace("T", "U")
    seq_m = seq_m.upper().replace("T", "U")
    h_counts = np.zeros(1024, dtype=np.float32)
    m_counts = np.zeros(1024, dtype=np.float32)
    for i in range(len(seq_h) - 4):
        km = seq_h[i:i+5]
        if km in kmer5_idx: h_counts[kmer5_idx[km]] += 1
    for i in range(len(seq_m) - 4):
        km = seq_m[i:i+5]
        if km in kmer5_idx: m_counts[kmer5_idx[km]] += 1
    # Normalise
    h_total = h_counts.sum()
    m_total = m_counts.sum()
    if h_total > 0: h_counts /= h_total
    if m_total > 0: m_counts /= m_total
    return h_counts - m_counts   # difference only


def local_context_features(seq_h, seq_m, window=5):
    """5 features from ±window nt around the SNP position."""
    seq_h = seq_h.upper().replace("T", "U")
    seq_m = seq_m.upper().replace("T", "U")
    snp_pos = next((i for i, (a, b) in enumerate(zip(seq_h, seq_m)) if a != b), -1)
    if snp_pos == -1:
        return np.zeros(5, dtype=np.float32)

    L = len(seq_h)
    start = max(0, snp_pos - window)
    end   = min(L, snp_pos + window + 1)
    ctx   = seq_h[start:end]

    ctx_gc   = sum(1 for b in ctx if b in "GC") / max(len(ctx), 1)
    ctx_len  = len(ctx) / (2 * window + 1)   # normalised
    ctx_au   = (ctx.count("AU") + ctx.count("UA")) / max(len(ctx) - 1, 1)
    ctx_gu   = (ctx.count("GU") + ctx.count("UG")) / max(len(ctx) - 1, 1)
    # Shannon entropy of 3-mer at SNP site
    trigram = seq_h[max(0, snp_pos-1): min(L, snp_pos+2)]
    counts  = Counter(trigram)
    total   = sum(counts.values())
    entropy = -sum((c/total) * _math.log2(c/total) for c in counts.values() if c > 0)

    return np.array([ctx_gc, ctx_len, ctx_au, ctx_gu, entropy], dtype=np.float32)


# ── Reuse seed_features and conservation_proxy from existing notebook ─────────
# These functions should already be defined in the notebook scope.
# If not, they are included here as fallback:

try:
    _ = seed_features
    print("Using existing seed_features function")
except NameError:
    print("WARNING: seed_features not found — using dummy (all zeros)")
    def seed_features(mirna_id, seq_h, seq_m):
        return np.zeros(4, dtype=np.float32)

try:
    _ = conservation_proxy
    print("Using existing conservation_proxy function")
except NameError:
    print("WARNING: conservation_proxy not found — using dummy (all zeros)")
    def conservation_proxy(seq_h, snp_pos, struct_h=None):
        return np.zeros(3, dtype=np.float32)


def build_kmer_features_v2(row):
    """Extended k-mer features: original 971 + 5-mer diff (1024) + local context (5) = 2000 dims."""
    seq_h    = row["Seq_Healthy"].upper().replace("T", "U")
    seq_m    = row["Seq_Mutant"].upper().replace("T", "U")
    mirna_id = row["miRNA_ID"]

    # Original k-mer frequencies (k=3 and k=4) — 960 dims
    kmer_feats = []
    for k in [3, 4]:
        h = kmer_freq(seq_h, k)
        m = kmer_freq(seq_m, k)
        kmer_feats.extend([h, m, h - m])

    snp_pos = next((i for i, (a, b) in enumerate(zip(seq_h, seq_m)) if a != b), 0)
    L = len(seq_h)

    # Positional properties (4 dims)
    props = np.array([
        (seq_h.count('G') + seq_h.count('C')) / max(L, 1),
        L / 128.0,
        snp_pos / max(L, 1),
        abs(snp_pos - L/2) / max(L/2, 1)
    ], dtype=np.float32)

    # Seed region features (4 dims)
    seed_f = seed_features(mirna_id, seq_h, seq_m)

    # Conservation proxy features (3 dims)
    cons_f = conservation_proxy(seq_h, snp_pos)

    # NEW: 5-mer difference vector (1024 dims)
    fivemar_diff = get_5mer_diff(seq_h, seq_m)

    # NEW: Local context features (5 dims)
    local_ctx = local_context_features(seq_h, seq_m, window=5)

    return np.concatenate(kmer_feats + [props, seed_f, cons_f, fivemar_diff, local_ctx])


print("Building extended feature matrix (v2)...")
X_kmer_v2 = np.stack([build_kmer_features_v2(row) for _, row in df.iterrows()])
X_kmer_v2 = np.nan_to_num(X_kmer_v2, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X_kmer_v2.shape}  (expected ~2000 cols)")


# ═══ CELL B2 — Optuna XGBoost tuning (50 trials per fold) ═══
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    print(f"Optuna version: {optuna.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"], check=True)
    import optuna
    print(f"Optuna installed: {optuna.__version__}")

try:
    from xgboost import XGBClassifier
    print("XGBoost available")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"], check=True)
    from xgboost import XGBClassifier
    print("XGBoost installed")

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

sgkf_kmer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

kmer_v2_fold_metrics = []
kmer_v2_all_probs    = []
kmer_v2_all_labels   = []
kmer_v2_fold_models  = []
kmer_v2_best_params  = []

optuna.logging.set_verbosity(optuna.logging.WARNING)

for fold, (train_idx, test_idx) in enumerate(sgkf_kmer.split(X_kmer_v2, y, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — k-mer v2 + Optuna")
    print(f"{'='*55}")

    X_tr, y_tr = X_kmer_v2[train_idx], y[train_idx]
    X_te, y_te = X_kmer_v2[test_idx],  y[test_idx]
    groups_tr  = groups[train_idx]

    # Inner CV for Optuna — 3-fold StratifiedGroupKFold on train only
    inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42+fold)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'eval_metric':      'logloss',
            'use_label_encoder': False,
            'random_state':     42,
            'verbosity':        0,
        }
        # Inner 3-fold CV on train data only
        aucs = []
        for inner_tr, inner_val in inner_sgkf.split(X_tr, y_tr, groups_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[inner_tr], y_tr[inner_tr])
            p = clf.predict_proba(X_tr[inner_val])[:, 1]
            aucs.append(roc_auc_score(y_tr[inner_val], p))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_p = study.best_params
    print(f"  Best params: {best_p}")
    print(f"  Best inner AUC: {study.best_value:.4f}")
    kmer_v2_best_params.append(best_p)

    # Retrain on full outer train with best params
    final_params = {
        'n_estimators':     best_p['n_estimators'],
        'max_depth':        best_p['max_depth'],
        'learning_rate':    best_p['lr'],
        'subsample':        best_p['subsample'],
        'colsample_bytree': best_p['col'],
        'min_child_weight': best_p['mcw'],
        'eval_metric':      'logloss',
        'use_label_encoder': False,
        'random_state':     42,
        'verbosity':        0,
    }
    model = XGBClassifier(**final_params)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    kmer_v2_all_probs.extend(probs.tolist())
    kmer_v2_all_labels.extend(y_te.tolist())
    kmer_v2_fold_models.append(model)

    m = {
        "accuracy":  accuracy_score(y_te, preds),
        "auc":       roc_auc_score(y_te, probs),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall":    recall_score(y_te, preds, zero_division=0),
        "f1":        f1_score(y_te, preds, zero_division=0),
    }
    kmer_v2_fold_metrics.append(m)
    print(f"  Fold {fold+1} Test — AUC: {m['auc']:.4f}  Acc: {m['accuracy']:.4f}  "
          f"F1: {m['f1']:.4f}")

    # Save fold model
    joblib.dump(model, f"DeepFold_models/kmer_xgb_v2_fold{fold}.pkl")

print(f"\n{'='*55}")
print("k-mer XGBoost v2 — 5-FOLD CV SUMMARY")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in kmer_v2_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nv1 AUC was 0.704 ± 0.021 — check improvement above.")
print(f"OOF AUC: {roc_auc_score(kmer_v2_all_labels, kmer_v2_all_probs):.4f}")
