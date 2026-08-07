import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib, os

CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
CKPT_DIR = "DeepFold_models"
os.makedirs(CKPT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def kmer_freq(seq, k=3):
    bases = "ACGU"
    kmers = ["".join(p) for p in __import__("itertools").product(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    seq = seq.upper().replace("T", "U")
    counts = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_idx: counts[kmer_idx[kmer]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts

def build_features(row):
    seq_h = row["Seq_Healthy"]
    seq_m = row["Seq_Mutant"]
    h3 = kmer_freq(seq_h, 3); m3 = kmer_freq(seq_m, 3)
    return np.concatenate([h3, m3, h3 - m3])

def train_xgb():
    print("Extracting k-mer features...")
    X = np.stack([build_features(row) for _, row in df.iterrows()])
    y = df["Label"].values
    groups = df["miRNA_ID"].values
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    print(f"XGB Training on {len(train_idx)} samples...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, n_jobs=-1)
    model.fit(X[train_idx], y[train_idx])
    
    auc = roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1])
    print(f"XGB AUC: {auc:.4f}")
    
    joblib.dump(model, os.path.join(CKPT_DIR, "kmer_xgb_v2_fold1.pkl"))
    print("Model saved: kmer_xgb_v2_fold1.pkl")

if __name__ == "__main__":
    train_xgb()
