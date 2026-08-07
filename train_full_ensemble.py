import os
import random
import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import optuna
from scipy.stats import rankdata

# --- CONFIG ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
NPY_DIR = "DeepFold_Dataset/processed_maps/npy"
MODELS_DIR = "DeepFold_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
# Ensure we only use samples that have corresponding NPY files
df = df[df["Sample_ID"].apply(lambda x: os.path.exists(os.path.join(NPY_DIR, f"{x}.npy")))].reset_index(drop=True)
print(f"Total samples with NPY files: {len(df)}")

# --- UTILS ---
def get_rank_probs(probs):
    return rankdata(probs) / len(probs)

# --- 1. CNN ARCHITECTURE (v4 with CoordAttention) ---

class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * scale

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch)
        self.project = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout2d(dropout_p)
    def forward(self, x): return self.drop(self.se(self.conv(x)) + self.project(x))

class MultiScalePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.AdaptiveAvgPool2d(1)
        self.p2 = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x)], dim=1)

class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_w * a_h

class DeepFoldCNN_v4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Split input paths
        self.main_path = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.ctx_path = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.input_attn = CoordAttention(48, 48)
        self.input_drop = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d(2) # 128 -> 64
        self.block2 = ResidualBlock(48, 64, 0.15)
        self.pool2 = nn.MaxPool2d(2) # 64 -> 32
        self.block3 = ResidualBlock(64, 128, 0.20)
        self.pool3 = nn.MaxPool2d(2) # 32 -> 16
        self.block4 = ResidualBlock(128, 256, 0.10)
        self.pool4 = nn.MaxPool2d(2) # 16 -> 8
        self.pool_final = MultiScalePool() # 512 channels
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        # x: [B, 4, 128, 128]
        # C1, C2 context path | C3, C4 main path
        xm = self.main_path(x[:, 2:4, :, :])
        xc = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([xm, xc], dim=1)
        x = self.pool1(self.input_drop(self.input_attn(x)))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return self.classifier(self.flatten(self.pool_final(x)))

# --- 2. GCN ARCHITECTURE ---

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, h, adj):
        return F.relu(self.norm(torch.bmm(adj, self.W(h))))

class SiameseGCN(nn.Module):
    def __init__(self, in_dim=6, hidden=64, embed_dim=128, dropout=0.3, num_classes=2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
    def encode(self, nf, adj, mask):
        h = self.drop(self.gcn1(nf, adj))
        h = self.drop(self.gcn2(h, adj))
        h = self.gcn3(h, adj)
        mask_exp = mask.unsqueeze(-1)
        return (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
    def forward(self, nf_h, adj_h, mask_h, nf_m, adj_m, mask_m):
        h = self.encode(nf_h, adj_h, mask_h)
        m = self.encode(nf_m, adj_m, mask_m)
        return self.classifier(torch.cat([h, m, torch.abs(h-m), h*m], dim=1))

# --- DATASETS ---

class SNPDataset_CNN(Dataset):
    def __init__(self, dataframe, augment=False):
        self.records = dataframe.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        t = np.load(os.path.join(NPY_DIR, f"{row['Sample_ID']}.npy")).astype(np.float32)
        # Channels: 0:healthy_map, 1:mutant_map, 2:diff, 3:mask
        t = torch.tensor(t).permute(2, 0, 1) # [C, H, W]
        if self.augment:
            if random.random() > 0.5: t = torch.flip(t, dims=[1])
            if random.random() > 0.5: t = torch.flip(t, dims=[2])
        label = torch.tensor(row["class"], dtype=torch.long)
        return t, label

def build_graph_tensors(seq, contact_map, threshold=0.5):
    base_map = {'A': 0, 'U': 1, 'T': 1, 'C': 2, 'G': 3}
    L = min(len(seq), 128)
    nf = np.zeros((128, 6), dtype=np.float32)
    for i in range(L):
        b = seq[i]
        if b in base_map: nf[i, base_map[b]] = 1.0
        nf[i, 4] = i / max(L-1, 1)
        nf[i, 5] = 1.0
    adj = np.zeros((128, 128), dtype=np.float32)
    adj[:L, :L] = (contact_map[:L, :L] > threshold).astype(np.float32)
    for i in range(L-1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    # Norm adj
    deg = adj.sum(axis=1, keepdims=True).clip(min=1)
    deg_inv = 1.0 / np.sqrt(deg)
    adj = deg_inv * adj * deg_inv.T
    mask = np.zeros(128, dtype=np.float32)
    mask[:L] = 1.0
    return nf, adj, mask

class RNAGraphDataset(Dataset):
    def __init__(self, dataframe):
        self.records = dataframe.reset_index(drop=True)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        t = np.load(os.path.join(NPY_DIR, f"{row['Sample_ID']}.npy")).astype(np.float32)
        nf_h, adj_h, mask_h = build_graph_tensors(row["Seq_Healthy"], t[:,:,0])
        nf_m, adj_m, mask_m = build_graph_tensors(row["Seq_Mutant"], t[:,:,1])
        label = torch.tensor(row["class"], dtype=torch.long)
        return torch.tensor(nf_h), torch.tensor(adj_h), torch.tensor(mask_h), \
               torch.tensor(nf_m), torch.tensor(adj_m), torch.tensor(mask_m), label

# --- TRAINING LOOP ---

def train_cnn_fold(fold, train_df, val_df, test_df):
    train_dl = DataLoader(SNPDataset_CNN(train_df, augment=True), batch_size=32, shuffle=True)
    val_dl = DataLoader(SNPDataset_CNN(val_df), batch_size=32, shuffle=False)
    test_dl = DataLoader(SNPDataset_CNN(test_df), batch_size=32, shuffle=False)
    
    model = DeepFoldCNN_v4().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    save_path = os.path.join(MODELS_DIR, f"cnn_v4_fold{fold}.pt")
    if os.path.exists(save_path):
        print(f"Skipping CNN Fold {fold} (Exists)")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        t_probs = []
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(DEVICE)
                out = model(x)
                sm = F.softmax(out, dim=1)[:, 1]
                t_probs.extend(sm.cpu().numpy())
        return t_probs
    
    best_auc = 0
    for epoch in range(10): # Reduced epochs for speed
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(DEVICE)
                out = model(x)
                sm = F.softmax(out, dim=1)[:, 1]
                probs.extend(sm.cpu().numpy())
                labels.extend(y.numpy())
        auc = roc_auc_score(labels, probs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            
    # Load best and predict on test
    model.load_state_dict(torch.load(save_path))
    model.eval()
    t_probs = []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            out = model(x)
            sm = F.softmax(out, dim=1)[:, 1]
            t_probs.extend(sm.cpu().numpy())
    return t_probs

def train_gcn_fold(fold, train_df, val_df, test_df):
    train_dl = DataLoader(RNAGraphDataset(train_df), batch_size=16, shuffle=True)
    val_dl = DataLoader(RNAGraphDataset(val_df), batch_size=16, shuffle=False)
    test_dl = DataLoader(RNAGraphDataset(test_df), batch_size=16, shuffle=False)
    
    model = SiameseGCN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_path = os.path.join(MODELS_DIR, f"gcn_fold{fold}.pt")
    if os.path.exists(save_path):
        print(f"Skipping GCN Fold {fold} (Exists)")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        t_probs = []
        with torch.no_grad():
            for *x, y in test_dl:
                x = [v.to(DEVICE) for v in x]
                out = model(*x)
                sm = F.softmax(out, dim=1)[:, 1]
                t_probs.extend(sm.cpu().numpy())
        return t_probs
    
    best_auc = 0
    for epoch in range(10): # Reduced epochs
        model.train()
        for *x, y in train_dl:
            x = [v.to(DEVICE) for v in x]
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(*x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for *x, y in val_dl:
                x = [v.to(DEVICE) for v in x]
                out = model(*x)
                sm = F.softmax(out, dim=1)[:, 1]
                probs.extend(sm.cpu().numpy())
                labels.extend(y.numpy())
        auc = roc_auc_score(labels, probs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            
    model.load_state_dict(torch.load(save_path))
    model.eval()
    t_probs = []
    with torch.no_grad():
        for *x, y in test_dl:
            x = [v.to(DEVICE) for v in x]
            out = model(*x)
            sm = F.softmax(out, dim=1)[:, 1]
            t_probs.extend(sm.cpu().numpy())
    return t_probs

# XGBoost related
def get_kmer_feats(seq, k_list=[3,4,5]):
    feats = []
    bases = 'ATCG'
    for k in k_list:
        kmers = {}
        for i in range(len(seq)-k+1):
            km = seq[i:i+k]
            kmers[km] = kmers.get(km, 0) + 1
        # To keep feature count reasonable, just use a few or hash
        # Standard k-mer feature extraction normally would use all combinations
        # Here we use a simplified version for small dataset speed
        feats.append(float(len(kmers)) / len(seq))
    return np.array(feats)

def train_xgb_models(X_train, y_train, X_test, model_name, fold):
    save_path = os.path.join(MODELS_DIR, f"{model_name}_fold{fold}.pkl")
    if os.path.exists(save_path):
        print(f"Skipping {model_name} Fold {fold} (Exists)")
        clf = joblib.load(save_path)
        return clf.predict_proba(X_test)[:, 1]
    
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=SEED)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    joblib.dump(clf, save_path)
    return probs

# --- MAIN EXECUTION ---
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups = df["mirna_id"].values
y_true = df["class"].values

oof_predictions = {
    "cnn": np.zeros(len(df)),
    "gcn": np.zeros(len(df)),
    "kmer": np.zeros(len(df)),
    "mfe": np.zeros(len(df)),
    "rna-fm": np.zeros(len(df))
}

# Simplified feature extraction for XGB
print("Extracting features for XGBoost...")
X_kmer = np.stack([get_kmer_feats(row["Seq_Mutant"]) for _, row in df.iterrows()])
# Dummy MFE and RNA-FM features if not available
X_mfe = np.random.randn(len(df), 5) # Placeholder for MFE logic
X_rnafm = np.random.randn(len(df), 10) # Placeholder for RNA-FM logic

for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, y_true, groups), 1):
    print(f"Processing Fold {fold}...")
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    # Val split (simplified)
    val_idx_in_train = np.random.choice(len(train_df), int(0.1*len(train_df)), replace=False)
    val_df = train_df.iloc[val_idx_in_train]
    train_df_eff = train_df.drop(train_df.index[val_idx_in_train])
    
    # CNN
    oof_predictions["cnn"][test_idx] = train_cnn_fold(fold, train_df_eff, val_df, test_df)
    # GCN
    oof_predictions["gcn"][test_idx] = train_gcn_fold(fold, train_df_eff, val_df, test_df)
    # K-mer XGB
    oof_predictions["kmer"][test_idx] = train_xgb_models(X_kmer[train_idx], y_true[train_idx], X_kmer[test_idx], "kmer_xgb", fold)
    # MFE XGB
    oof_predictions["mfe"][test_idx] = train_xgb_models(X_mfe[train_idx], y_true[train_idx], X_mfe[test_idx], "mfe_xgb", fold)
    # RNA-FM XGB
    oof_predictions["rna-fm"][test_idx] = train_xgb_models(X_rnafm[train_idx], y_true[train_idx], X_rnafm[test_idx], "rnafm_xgb", fold)

# Meta-Learner
print("Training Meta-Learner...")
# Rank normalise OOF
X_meta = np.stack([get_rank_probs(oof_predictions[m]) for m in oof_predictions], axis=1)
meta_clf = xgb.XGBClassifier(n_estimators=50, max_depth=2, random_state=SEED)
meta_clf.fit(X_meta, y_true)
joblib.dump(meta_clf, os.path.join(MODELS_DIR, "meta_learner_v2.pkl"))

# Final Summary Table
print("\n" + "="*40)
print("FINAL ENSEMBLE SUMMARY")
print("="*40)
for m in oof_predictions:
    auc = roc_auc_score(y_true, oof_predictions[m])
    print(f"{m:10}: AUC = {auc:.4f}")
meta_auc = roc_auc_score(y_true, meta_clf.predict_proba(X_meta)[:, 1])
print(f"{'Meta':10}: AUC = {meta_auc:.4f}")
print("="*40)

# Save Registry Config
config = {
    "auc_scores": {m: float(roc_auc_score(y_true, oof_predictions[m])) for m in oof_predictions},
    "meta_auc": float(meta_auc),
    "model_names": list(oof_predictions.keys()),
    "threshold": 0.5 # Default
}
joblib.dump(config, os.path.join(MODELS_DIR, "pipeline_config.pkl"))
print("Saved DeepFold Registry Config.")
