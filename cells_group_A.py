# ═══════════════════════════════════════════════════════════════════════════════
# CELL GROUP A — DeepFoldCNN v4
# Copy each "# ═══ CELL ═══" section into a separate Jupyter notebook cell.
# These cells go AFTER all existing cells in final.ipynb.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══ CELL A0 — Markdown ═══
# ---
# ## Step 10 — DeepFoldCNN v4 (Accuracy Improvements)
#
# ### Changes from v3
# - **Multi-scale pooling**: Replaces single AdaptiveAvgPool2d(4) with global avg + global max + 2×2 spatial pool → 1536 dims (was 4096)
# - **Coordinate Attention**: Position-aware attention after the split-path merge layer
# - **Label smoothing** (ε=0.10): Prevents overconfidence on noisy COSMIC labels
# - **Mixup augmentation** (α=0.2): Interpolates contact map tensors for smoother decision boundaries


# ═══ CELL A1 — v4 Architecture ═══
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import os, random, math

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
NPY_DIR  = "DeepFold_Dataset/processed_maps/npy"
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['Label'].value_counts().to_dict()}")


class SNPDataset_v4(Dataset):
    """v4 dataset — identical to v3 but returns one-hot labels for mixup."""
    def __init__(self, records, augment=False, num_classes=2):
        self.records = records.reset_index(drop=True)
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row    = self.records.iloc[idx]
        tensor = np.load(f"{NPY_DIR}/{row['Sample_ID']}.npy")
        tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1)

        if self.augment:
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2])

        label_idx = torch.tensor(row["Label"], dtype=torch.long)
        # One-hot for mixup compatibility
        label_oh = torch.zeros(self.num_classes, dtype=torch.float32)
        label_oh[row["Label"]] = 1.0
        return tensor, label_idx, label_oh, row["miRNA_ID"]


class MultiScalePool(nn.Module):
    """Replaces AdaptiveAvgPool2d(4) — captures global and local spatial info."""
    def forward(self, x):
        p1 = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, 256)
        p2 = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, 256)
        p3 = F.adaptive_avg_pool2d(x, 2).flatten(1)   # (B, 1024)
        return torch.cat([p1, p2, p3], dim=1)           # (B, 1536)


class CoordAttention(nn.Module):
    """Position-aware channel attention via horizontal + vertical pooling."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h_pool = x.mean(dim=3, keepdim=True)                    # (B,C,H,1)
        w_pool = x.mean(dim=2, keepdim=True).permute(0,1,3,2)   # (B,C,W,1) → (B,C,W,1)
        # Concatenate along H dimension: (B, C, H+W, 1)
        combined = torch.cat([h_pool, w_pool], dim=2)            # (B,C,H+W,1)
        att = torch.sigmoid(self.bn(self.conv(combined)))        # (B,C,H+W,1)
        att_h = att[:, :, :H, :]                                 # (B,C,H,1)
        att_w = att[:, :, H:, :].permute(0, 1, 3, 2)            # (B,C,1,W)
        return x * att_h * att_w


# SEBlock and ResidualBlock — identical to v3
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
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.se      = SEBlock(out_ch)
        self.drop    = nn.Dropout2d(dropout_p)
        self.project = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                        if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        residual = self.project(x)
        out = self.se(self.conv(x))
        return self.drop(out + residual)


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
print("v4 ready: CoordAttention + MultiScalePool + label_smoothing + mixup")


# ═══ CELL A2 — Training utilities (v4) ═══

def cosine_warmup_schedule(optimizer, warmup_epochs, total_epochs):
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

print("v4 training utilities defined (mixup, soft CE, label smoothing).")


# ═══ CELL A3 — v4 Training Loop (5-fold StratifiedGroupKFold) ═══

EPOCHS        = 120
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
print(f"\nv3 AUC was 0.625 ± 0.018 — check improvement above.")
