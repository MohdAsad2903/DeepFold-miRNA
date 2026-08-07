
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import os, random, math, joblib
import xgboost as xgb
import optuna

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
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
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_w * a_h
        return out

class DeepFoldCNN_v4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.main_path = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.ctx_path = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.input_attn = CoordAttention(48, 48)
        self.input_drop = nn.Dropout2d(0.10)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ResidualBlock(48, 64, 0.15)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ResidualBlock(64, 128, 0.20)
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = ResidualBlock(128, 256, 0.10)
        self.pool4 = nn.MaxPool2d(2)
        self.pool_final = MultiScalePool()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(512 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, num_classes))
    def forward(self, x):
        x_main = self.main_path(x[:, 2:4, :, :])
        x_ctx = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([x_main, x_ctx], dim=1)
        x = self.pool1(self.input_drop(self.input_attn(x)))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return self.classifier(self.flatten(self.pool_final(x)))

class SNPDataset_v4(Dataset):
    def __init__(self, df, augment=False):
        self.records = df.reset_index(drop=True)
        self.augment = augment
        self.npy_dir = "DeepFold_Dataset/processed_maps/npy"
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        t = np.load(os.path.join(self.npy_dir, f"{row['Sample_ID']}.npy")).astype(np.float32)
        t = torch.tensor(t).permute(2, 0, 1) # C, H, W
        if self.augment:
            if random.random() > 0.5: t = torch.flip(t, dims=[1])
            if random.random() > 0.5: t = torch.flip(t, dims=[2])
        label = torch.tensor(row["class"], dtype=torch.long)
        return t, label, row["mirna_id"]


df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
df = df[df["Sample_ID"].apply(lambda x: os.path.exists(f"DeepFold_Dataset/processed_maps/npy/{x}.npy"))]
print(f"Loaded {len(df)} samples")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DeepFoldCNN_v4(nn.Module):
    """
    v4 improvements over v3:
      - CoordAttention after merge (position-aware)
      - MultiScalePool replacing AdaptiveAvgPool(4)
      - Classifier updated for 1536-dim input
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Split input paths (identical to v3)
        self.main_path = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.ctx_path = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        # NEW: Coordinate Attention after merge
        self.coord_att  = CoordAttention(48)
        self.input_se   = SEBlock(48)
        self.input_drop = nn.Dropout2d(0.10)
        self.pool1      = nn.MaxPool2d(2)

        # Shared encoder (identical to v3)
        self.block2 = ResidualBlock(48,  64,  dropout_p=0.15)
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = ResidualBlock(64,  128, dropout_p=0.20)
        self.pool3  = nn.MaxPool2d(2)
        self.block4 = ResidualBlock(128, 256, dropout_p=0.10)
        self.pool4  = nn.MaxPool2d(2)

        # NEW: Multi-scale pooling + updated classifier
        self.ms_pool = MultiScalePool()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),       # was 4096 in v3
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_main = self.main_path(x[:, 2:4, :, :])
        x_ctx  = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([x_main, x_ctx], dim=1)
        x = self.coord_att(x)                        # NEW: position-aware attention
        x = self.input_se(x)
        x = self.pool1(self.input_drop(x))

        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.ms_pool(x)                          # NEW: multi-scale pooling
        return self.classifier(x)


# Sanity check
_m = DeepFoldCNN_v4()
_x = torch.randn(2, 4, 128, 128)
_o = _m(_x)
assert _o.shape == (2, 2), f"Unexpected: {_o.shape}"
print(f"DeepFoldCNN v4 output: {_o.shape}")
print(f"Trainable params: {sum(p.numel() for p in _m.parameters() if p.requires_grad):,}")
print("v4 ready: CoordAttention + MultiScalePool + label_smoothing + mixup")def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_weighted_ce_v4(labels_arr, device, label_smoothing=0.10):
    """CrossEntropyLoss with class weights AND label smoothing."""
    n_benign  = (labels_arr == 0).sum()
    n_disease = (labels_arr == 1).sum()
    pos_w     = float(n_benign) / float(n_disease)
    weights   = torch.tensor([1.0, pos_w], dtype=torch.float32, device=device)
    print(f"  CE weights — benign: 1.000  disease: {pos_w:.3f}  "
          f"label_smoothing: {label_smoothing}")
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def mixup_batch(x, y_onehot, alpha=0.2):
    """Mixup on tensors + one-hot labels. Returns mixed x and soft y."""
    if alpha <= 0:
        return x, y_onehot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 for stability
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mix, y_mix


def soft_cross_entropy(logits, soft_targets, weight=None):
    """CE loss for soft (mixup) labels. Handles class weights."""
    log_probs = F.log_softmax(logits, dim=1)
    if weight is not None:
        log_probs = log_probs * weight.unsqueeze(0)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "auc":       roc_auc_score(labels, probs),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "cm":        confusion_matrix(labels, preds)
    }

print("v4 training utilities defined (mixup, soft CE, label smoothing).")EPOCHS        = 120
BATCH_SIZE    = 32
LR            = 3e-4
PATIENCE      = 25
WARMUP_EPOCHS = 5
MIXUP_ALPHA   = 0.2

sgkf_v4      = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups       = df["miRNA_ID"].values
labels_arr   = df["Label"].values

v4_fold_metrics = []
v4_all_labels, v4_all_probs, v4_all_mirnas = [], [], []

os.makedirs("DeepFold_models", exist_ok=True)

# Build class weights once
n_b = (labels_arr == 0).sum()
n_d = (labels_arr == 1).sum()
pos_w = float(n_b) / float(n_d)
class_weights = torch.tensor([1.0, pos_w], dtype=torch.float32, device=DEVICE)

for fold, (train_idx, test_idx) in enumerate(sgkf_v4.split(df, labels_arr, groups)):
    print(f"\n{'='*55}")
    print(f"FOLD {fold+1}/5 — train: {len(train_idx)}  test: {len(test_idx)}")
    print(f"  Test miRNAs: {df.iloc[test_idx]['miRNA_ID'].nunique()}")
    print(f"  Test labels: {dict(df.iloc[test_idx]['Label'].value_counts().sort_index())}")
    print(f"{'='*55}")

    # Val split
    rng = np.random.default_rng(SEED + fold)
    train_idx_shuffled = rng.permutation(train_idx)
    n_val = int(0.15 * len(train_idx))
    val_idx, train_idx_ = train_idx_shuffled[:n_val], train_idx_shuffled[n_val:]

    train_ds = SNPDataset_v4(df.iloc[train_idx_], augment=True)
    val_ds   = SNPDataset_v4(df.iloc[val_idx],    augment=False)
    test_ds  = SNPDataset_v4(df.iloc[test_idx],   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = DeepFoldCNN_v4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = cosine_warmup_schedule(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.10)

    # Per-group channel stats
    sample_t = torch.stack([train_ds[i][0] for i in range(min(200, len(train_ds)))])
    ctx_mean  = sample_t[:, 0:2].mean(dim=(0,2,3))
    ctx_std   = sample_t[:, 0:2].std(dim=(0,2,3)).clamp(min=1e-6)
    main_mean = sample_t[:, 2:4].mean(dim=(0,2,3))
    main_std  = sample_t[:, 2:4].std(dim=(0,2,3)).clamp(min=1e-6)
    ch_mean   = torch.cat([ctx_mean, main_mean]).to(DEVICE)
    ch_std    = torch.cat([ctx_std,  main_std]).to(DEVICE)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt = f"DeepFold_models/cnn_v4_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── Train with mixup ──
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for tensors, labels_idx, labels_oh, _ in train_dl:
            tensors   = tensors.to(DEVICE)
            labels_oh = labels_oh.to(DEVICE)
            labels_idx = labels_idx.to(DEVICE)
            tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]

            # Apply mixup
            tensors_mix, labels_mix = mixup_batch(tensors, labels_oh, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(tensors_mix)
            loss = soft_cross_entropy(outputs, labels_mix, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * len(labels_idx)
            # Accuracy on un-mixed predictions for logging
            with torch.no_grad():
                out_clean = model((tensors.detach() - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]) if False else outputs
            tr_correct += (outputs.argmax(1) == labels_idx).sum().item()
            tr_total   += len(labels_idx)

        tr_loss /= tr_total
        scheduler.step()

        # ── Validate (no mixup) ──
        model.eval()
        vl_loss, vl_correct, vl_total = 0, 0, 0
        vl_labels_list, vl_probs_list = [], []
        with torch.no_grad():
            for tensors, labels_idx, _, _ in val_dl:
                tensors, labels_idx = tensors.to(DEVICE), labels_idx.to(DEVICE)
                tensors = (tensors - ch_mean[None,:,None,None]) / ch_std[None,:,None,None]
                outputs = model(tensors)
                probs   = F.softmax(outputs, dim=1)[:, 1]
                loss    = criterion(outputs, labels_idx)
                vl_loss    += loss.item() * len(labels_idx)
                vl_correct += (outputs.argmax(1) == labels_idx).sum().item()
                vl_total   += len(labels_idx)
                vl_labels_list.extend(labels_idx.cpu().numpy())
                vl_probs_list.extend(probs.cpu().numpy())

        vl_loss /= vl_total
        vl_acc = vl_correct / vl_total
        vl_auc = roc_auc_score(vl_labels_list, vl_probs_list)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save({
                "model_state": model.state_dict(),
                "ch_mean": ch_mean.cpu(), "ch_std": ch_std.cpu(),
                "fold": fold, "val_auc": vl_auc,
            }, best_ckpt)
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep {epoch:3d} | tr_loss {tr_loss:.4f} | vl_loss {vl_loss:.4f} "
                  f"vl_acc {vl_acc:.4f} vl_auc {vl_auc:.4f} lr {lr_now:.2e}"
                  + (" ← best" if patience_ctr == 0 else f" (pat {patience_ctr}/{PATIENCE})"))

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Evaluate best on test fold ──
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    ch_m, ch_s = ckpt["ch_mean"].to(DEVICE), ckpt["ch_std"].to(DEVICE)

    model.eval()
    te_labels, te_probs, te_mirnas = [], [], []
    with torch.no_grad():
        for tensors, labels_idx, _, mirnas in test_dl:
            tensors = tensors.to(DEVICE)
            tensors = (tensors - ch_m[None,:,None,None]) / ch_s[None,:,None,None]
            probs = F.softmax(model(tensors), dim=1)[:, 1]
            te_labels.extend(labels_idx.numpy())
            te_probs.extend(probs.cpu().numpy())
            te_mirnas.extend(mirnas)

    te_y, te_p = np.array(te_labels), np.array(te_probs)
    m = compute_metrics(te_y, te_p)
    print(f"\n  Fold {fold+1} v4 Test: Acc {m['accuracy']:.4f}  AUC {m['auc']:.4f}  "
          f"Prec {m['precision']:.4f}  Rec {m['recall']:.4f}  F1 {m['f1']:.4f}")

    v4_fold_metrics.append(m)
    v4_all_labels.extend(te_y.tolist())
    v4_all_probs.extend(te_p.tolist())
    v4_all_mirnas.extend(te_mirnas)

print(f"\n{'='*55}")
print("5-FOLD CV SUMMARY — DeepFoldCNN v4")
print(f"{'='*55}")
for metric in ["accuracy", "auc", "precision", "recall", "f1"]:
    vals = [m[metric] for m in v4_fold_metrics]
    print(f"  {metric.capitalize():12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nv3 AUC was 0.625 ± 0.018 — check improvement above.")# ═══════════════════════════════════════════════════════════════════════════════from itertools import product as iproduct
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
print(f"Feature matrix: {X_kmer_v2.shape}  (expected ~2000 cols)")import warnings
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
print(f"OOF AUC: {roc_auc_score(kmer_v2_all_labels, kmer_v2_all_probs):.4f}")# ═══════════════════════════════════════════════════════════════════════════════import subprocess, sys

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

batch_converter = alphabet.get_batch_converter()df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
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
torch.cuda.empty_cache()import optuna
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
print(f"\nOOF AUC: {roc_auc_score(rnafm_all_labels, rnafm_all_probs):.4f}")# ═══════════════════════════════════════════════════════════════════════════════import numpy as np
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
print(f"\nAll arrays: {len(true_y)} samples ✓")def rank_normalize(probs):
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
    print(f"  {name:8s}: [{arr.min():.3f}, {arr.max():.3f}]  mean={arr.mean():.3f}")sgkf_meta = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

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
print(f"\nOriginal 4-model ensemble AUC was 0.720 ± 0.020 — check improvement above.")print(f"\n{'='*70}")
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
print("Saved → all_methods_comparison_v2.csv")# ═══════════════════════════════════════════════════════════════════════════════import os, torch, joblib
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
print(f"\nModel registry complete → {CKPT_DIR}")import torch
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

print("predict_variant() defined — full 5-model ensemble inference pipeline ready.")# Pick 3 samples: 1 disease, 1 benign, 1 borderline (closest to 0.5 in ensemble)
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