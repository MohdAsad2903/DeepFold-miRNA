import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os

CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
NPY_DIR = "DeepFold_Dataset/processed_maps/npy"
CKPT_DIR = "DeepFold_models"
DEVICE = torch.device("cpu")
MAX_L = 128

df = pd.read_csv(CSV_PATH)

def build_graph_tensors(seq, contact_map, threshold=0.5):
    base_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'T': 1}
    seq = seq.upper()
    L = min(len(seq), MAX_L)
    node_feat = np.zeros((MAX_L, 6), dtype=np.float32)
    for i in range(L):
        if seq[i] in base_map: node_feat[i, base_map[seq[i]]] = 1.0
        node_feat[i, 4] = i / max(L - 1, 1)
        node_feat[i, 5] = 1.0
    adj = np.zeros((MAX_L, MAX_L), dtype=np.float32)
    cm = contact_map[:L, :L]
    adj[:L, :L] = (cm > threshold).astype(np.float32)
    for i in range(L - 1):
        adj[i, i+1] = 1.0
        adj[i+1, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1, keepdims=True).clip(min=1)
    deg_inv = 1.0 / np.sqrt(deg)
    adj = deg_inv * adj * deg_inv.T
    mask = np.zeros(MAX_L, dtype=np.float32)
    mask[:L] = 1.0
    return node_feat, adj, mask

class RNAGraphDataset(Dataset):
    def __init__(self, records):
        self.records = records.reset_index(drop=True)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        npy_path = os.path.join(NPY_DIR, f"{row['Sample_ID']}.npy")
        t = np.load(npy_path)
        nf_h, adj_h, mask_h = build_graph_tensors(row["Seq_Healthy"], t[:,:,0])
        nf_m, adj_m, mask_m = build_graph_tensors(row["Seq_Mutant"],  t[:,:,1])
        return (torch.tensor(nf_h), torch.tensor(adj_h), torch.tensor(mask_h),
                torch.tensor(nf_m), torch.tensor(adj_m), torch.tensor(mask_m),
                torch.tensor(row["Label"], dtype=torch.long))

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_ch := out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, h, adj):
        return F.relu(self.norm(torch.bmm(adj, self.W(h))))

class SiameseGCN(nn.Module):
    def __init__(self, in_dim=6, hidden=64, embed_dim=128):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim * 4, 128), nn.ReLU(), nn.Linear(128, 2))
    def encode(self, nf, adj, mask):
        h = self.gcn1(nf, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        mask_exp = mask.unsqueeze(-1)
        return (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
    def forward(self, nf_h, adj_h, mask_h, nf_m, adj_m, mask_m):
        h = self.encode(nf_h, adj_h, mask_h)
        m = self.encode(nf_m, adj_m, mask_m)
        return self.classifier(torch.cat([h, m, torch.abs(h-m), h*m], dim=1))

def train_gcn():
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, df["Label"], df["miRNA_ID"]))
    train_dl = DataLoader(RNAGraphDataset(df.iloc[train_idx]), batch_size=32, shuffle=True)
    test_dl = DataLoader(RNAGraphDataset(df.iloc[test_idx]), batch_size=32, shuffle=False)
    model = SiameseGCN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 11): # reduced epochs
        model.train()
        for nf_h, adj_h, mask_h, nf_m, adj_m, mask_m, labels in train_dl:
            optimizer.zero_grad()
            out = model(nf_h.to(DEVICE), adj_h.to(DEVICE), mask_h.to(DEVICE), nf_m.to(DEVICE), adj_m.to(DEVICE), mask_m.to(DEVICE))
            loss = criterion(out, labels.to(DEVICE))
            loss.backward(); optimizer.step()
        print(f"GCN Epoch {epoch} complete")
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "gcn_fold1_best.pt"))
    print("Model saved: gcn_fold1_best.pt")

if __name__ == "__main__":
    present = len([f for f in os.listdir(NPY_DIR) if f.endswith('.npy')])
    if present > 1000:
        train_gcn()
    else:
        print("Waiting for more data for GCN...")
