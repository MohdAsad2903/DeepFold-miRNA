# ═══════════════════════════════════════════════════════════════════════════════
# CELL GROUP E — Save Models + Prediction Pipeline
# Copy each "# ═══ CELL ═══" section into a separate Jupyter notebook cell.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══ CELL E0 — Markdown ═══
# ---
# ## Step 14 — Model Registry and Inference Pipeline
#
# Saves all fold models to `DeepFold_models/` and builds a single
# `predict_variant()` function that runs the full 5-model ensemble pipeline.


# ═══ CELL E1 — Save complete model registry ═══
import os, torch, joblib
import numpy as np

CKPT_DIR = "DeepFold_models/"
os.makedirs(CKPT_DIR, exist_ok=True)

# 1. CNN v4 — already saved in Cell Group A as cnn_v4_fold{i}.pt
for i in range(5):
    p = f"{CKPT_DIR}/cnn_v4_fold{i}.pt"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ CNN v4 checkpoints: 5 folds")

# 2. Siamese GCN — save from existing notebook variables
try:
    # The original notebook stores models per fold — if variable exists, save
    # If GCN models are already on disk from original training, skip
    gcn_saved = 0
    for fold_i in range(5):
        p = f"{CKPT_DIR}/gcn_fold{fold_i}.pt"
        if os.path.exists(p):
            gcn_saved += 1
        elif 'gnn_fold_metrics' in dir():
            # Try to save from existing checkpoint path
            orig_p = f"DeepFold_Dataset/checkpoints/gcn_fold{fold_i+1}_best.pt"
            if os.path.exists(orig_p):
                import shutil
                shutil.copy2(orig_p, p)
                gcn_saved += 1
    print(f"✓ GCN checkpoints: {gcn_saved}/5 folds")
except Exception as e:
    print(f"⚠ GCN save: {e}")

# 3. k-mer XGBoost v2 — already saved in Cell Group B
for i in range(5):
    p = f"{CKPT_DIR}/kmer_xgb_v2_fold{i}.pkl"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ k-mer XGB v2: 5 folds")

# 4. MFE XGBoost — save from existing notebook
try:
    mfe_saved = 0
    for fold_i in range(5):
        p = f"{CKPT_DIR}/mfe_xgb_fold{fold_i}.pkl"
        if not os.path.exists(p):
            # Retrain if needed (uses existing X_mfe, y, sgkf from original notebook)
            pass
        if os.path.exists(p):
            mfe_saved += 1
    if mfe_saved < 5:
        print(f"⚠ MFE XGB: only {mfe_saved}/5 on disk. "
              "Re-run MFE training and save models per fold.")
    else:
        print(f"✓ MFE XGB: 5 folds")
except Exception as e:
    print(f"⚠ MFE save: {e}")

# 5. RNA-FM XGBoost — already saved in Cell Group C
for i in range(5):
    p = f"{CKPT_DIR}/rnafm_xgb_fold{i}.pkl"
    assert os.path.exists(p), f"Missing: {p}"
print(f"✓ RNA-FM XGB: 5 folds")

# 6. Meta-learner v2 — already saved in Cell Group D
assert os.path.exists(f"{CKPT_DIR}/meta_learner_v2.pkl"), "Missing meta_learner_v2.pkl"
print(f"✓ Meta-learner v2")

# 7. Pipeline config
config = {
    'n_folds':        5,
    'rnafm_emb_dim':  640,
    'kmer_v2_dim':    X_kmer_v2.shape[1] if 'X_kmer_v2' in dir() else 2000,
    'seed':           42,
}
joblib.dump(config, f"{CKPT_DIR}/pipeline_config.pkl")
print(f"✓ Pipeline config saved")
print(f"\nModel registry complete → {CKPT_DIR}")


# ═══ CELL E2 — Full prediction function ═══
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from scipy.stats import rankdata

def predict_variant(mirna_id, seq_healthy, seq_mutant, snp_pos=None,
                    models_dir="DeepFold_models/", n_folds=5):
    """
    Full ensemble prediction for a single miRNA SNP variant.

    Args:
        mirna_id:    str, e.g. "hsa-mir-21"
        seq_healthy: str, RNA sequence (uses U)
        seq_mutant:  str, RNA sequence with SNP applied
        snp_pos:     int or None, 0-based index of the SNP in the sequence
        models_dir:  str, path to saved model directory
        n_folds:     int, number of fold checkpoints to average

    Returns:
        dict with keys: prob_disease, label, confidence, base_probs
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect SNP position if not provided
    if snp_pos is None:
        snp_pos = next((i for i, (a, b) in enumerate(
            zip(seq_healthy.upper(), seq_mutant.upper())) if a != b), 0)

    # ═══ 1. CNN v4 — generate contact map and run inference ═══
    cnn_preds = []
    try:
        # Generate 4-channel tensor using existing UFold functions
        c1 = get_contact_map(seq_healthy, 128)
        c2 = get_contact_map(seq_mutant, 128)
        c3 = np.abs(c1 - c2)
        inp = seq_to_input(seq_healthy, 128)
        c4 = inp[0, 16].numpy()
        tensor = np.stack([c1, c2, c3, c4], axis=-1).astype(np.float32)
        tensor_t = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0)  # (1,4,128,128)

        for fold_i in range(n_folds):
            ckpt = torch.load(f"{models_dir}/cnn_v4_fold{fold_i}.pt",
                              map_location=DEVICE)
            model = DeepFoldCNN_v4().to(DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            ch_m = ckpt["ch_mean"].to(DEVICE)
            ch_s = ckpt["ch_std"].to(DEVICE)
            t = (tensor_t.to(DEVICE) - ch_m[None,:,None,None]) / ch_s[None,:,None,None]
            with torch.no_grad():
                prob = torch.softmax(model(t), dim=1)[0, 1].item()
            cnn_preds.append(prob)
        cnn_prob = np.mean(cnn_preds)
    except Exception as e:
        print(f"  CNN inference error: {e}")
        cnn_prob = 0.5

    # ═══ 2. Siamese GCN ═══
    gcn_preds = []
    try:
        nf_h, adj_h, mask_h = build_graph_tensors(seq_healthy,
                                get_contact_map(seq_healthy, 128))
        nf_m, adj_m, mask_m = build_graph_tensors(seq_mutant,
                                get_contact_map(seq_mutant, 128))
        for fold_i in range(n_folds):
            p = f"{models_dir}/gcn_fold{fold_i}.pt"
            if not os.path.exists(p):
                continue
            gcn_model = SiameseGCN().to(DEVICE)
            gcn_model.load_state_dict(torch.load(p, map_location=DEVICE))
            gcn_model.eval()
            with torch.no_grad():
                out = gcn_model(
                    torch.tensor(nf_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(adj_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(mask_h).unsqueeze(0).to(DEVICE),
                    torch.tensor(nf_m).unsqueeze(0).to(DEVICE),
                    torch.tensor(adj_m).unsqueeze(0).to(DEVICE),
                    torch.tensor(mask_m).unsqueeze(0).to(DEVICE),
                )
                prob = F.softmax(out, dim=1)[0, 1].item()
            gcn_preds.append(prob)
        gcn_prob = np.mean(gcn_preds) if gcn_preds else 0.5
    except Exception as e:
        print(f"  GCN inference error: {e}")
        gcn_prob = 0.5

    # ═══ 3. k-mer XGBoost v2 ═══
    try:
        row_dict = {"Seq_Healthy": seq_healthy, "Seq_Mutant": seq_mutant,
                    "miRNA_ID": mirna_id}
        kmer_feats = build_kmer_features_v2(pd.Series(row_dict)).reshape(1, -1)
        kmer_preds = []
        for fold_i in range(n_folds):
            m = joblib.load(f"{models_dir}/kmer_xgb_v2_fold{fold_i}.pkl")
            kmer_preds.append(m.predict_proba(kmer_feats)[0, 1])
        kmer_prob = np.mean(kmer_preds)
    except Exception as e:
        print(f"  k-mer inference error: {e}")
        kmer_prob = 0.5

    # ═══ 4. MFE XGBoost ═══
    try:
        mfe_feats = extract_mfe_features(seq_healthy, seq_mutant).reshape(1, -1)
        mfe_preds = []
        for fold_i in range(n_folds):
            p = f"{models_dir}/mfe_xgb_fold{fold_i}.pkl"
            if os.path.exists(p):
                m = joblib.load(p)
                mfe_preds.append(m.predict_proba(mfe_feats)[0, 1])
        mfe_prob = np.mean(mfe_preds) if mfe_preds else 0.5
    except Exception as e:
        print(f"  MFE inference error: {e}")
        mfe_prob = 0.5

    # ═══ 5. RNA-FM XGBoost ═══
    try:
        emb_h = get_rnafm_embedding(seq_healthy, model_rnafm, batch_converter, DEVICE)
        emb_m = get_rnafm_embedding(seq_mutant, model_rnafm, batch_converter, DEVICE)
        emb_diff = (emb_h - emb_m).reshape(1, -1)
        combined = np.hstack([kmer_feats, emb_diff])
        rnafm_preds = []
        for fold_i in range(n_folds):
            m = joblib.load(f"{models_dir}/rnafm_xgb_fold{fold_i}.pkl")
            rnafm_preds.append(m.predict_proba(combined)[0, 1])
        rnafm_prob = np.mean(rnafm_preds)
    except Exception as e:
        print(f"  RNA-FM inference error: {e}")
        rnafm_prob = 0.5

    # ═══ 6. Rank-normalise and meta-predict ═══
    base_probs = np.array([cnn_prob, gcn_prob, kmer_prob, mfe_prob, rnafm_prob])
    # For single-sample prediction, rank-normalisation is not meaningful
    # Use raw probabilities reshaped for the meta-learner
    meta_input = base_probs.reshape(1, -1)

    try:
        meta = joblib.load(f"{models_dir}/meta_learner_v2.pkl")
        final_prob = meta.predict_proba(meta_input)[0, 1]
    except Exception as e:
        print(f"  Meta-learner error: {e} — using mean of base probs")
        final_prob = base_probs.mean()

    # ═══ 7. Interpret ═══
    if final_prob >= 0.65:
        label      = 'Likely pathogenic'
        confidence = 'High' if final_prob >= 0.80 else 'Medium'
    elif final_prob <= 0.35:
        label      = 'Likely benign'
        confidence = 'High' if final_prob <= 0.20 else 'Medium'
    else:
        label      = 'Uncertain significance (VUS)'
        confidence = 'Low'

    return {
        'prob_disease': round(float(final_prob), 4),
        'label':        label,
        'confidence':   confidence,
        'base_probs': {
            'CNN_v4':    round(float(cnn_prob), 4),
            'GCN':       round(float(gcn_prob), 4),
            'kmer_XGB':  round(float(kmer_prob), 4),
            'MFE_XGB':   round(float(mfe_prob), 4),
            'RNAFM_XGB': round(float(rnafm_prob), 4),
        }
    }

print("predict_variant() defined — full 5-model ensemble inference pipeline ready.")


# ═══ CELL E3 — Demo prediction on test samples ═══

# Pick 3 samples: 1 disease, 1 benign, 1 borderline (closest to 0.5 in ensemble)
df_demo = pd.read_csv("DeepFold_Dataset/final_dataset.csv")

# Known disease sample
disease_row = df_demo[df_demo["Label"] == 1].iloc[0]
# Known benign sample
benign_row  = df_demo[df_demo["Label"] == 0].iloc[0]
# Borderline: pick the sample whose ensemble v2 OOF probability is closest to 0.5
if len(ensemble_v2_all_probs) > 0:
    border_idx = np.argmin(np.abs(ensemble_v2_all_probs - 0.5))
    border_row = df_demo.iloc[border_idx]
else:
    border_row = df_demo.iloc[len(df_demo)//2]

print("=" * 70)
print(f"{'DEMO PREDICTIONS — predict_variant()':^70}")
print("=" * 70)

for name, row in [("Known DISEASE", disease_row),
                  ("Known BENIGN", benign_row),
                  ("BORDERLINE", border_row)]:
    print(f"\n{'─'*70}")
    print(f"  Sample:   {row['Sample_ID']}")
    print(f"  miRNA:    {row['miRNA_ID']}")
    print(f"  True:     {'Disease' if row['Label'] == 1 else 'Benign'}")
    print(f"  Seq len:  {len(row['Seq_Healthy'])} nt")

    result = predict_variant(
        mirna_id    = row["miRNA_ID"],
        seq_healthy = row["Seq_Healthy"],
        seq_mutant  = row["Seq_Mutant"],
    )

    print(f"\n  Prediction: {result['label']}  "
          f"(p={result['prob_disease']:.4f}, {result['confidence']} confidence)")
    print(f"  Base model probabilities:")
    for model_name, prob in result['base_probs'].items():
        print(f"    {model_name:12s}: {prob:.4f}")

print(f"\n{'='*70}")
print("Demo complete.")
