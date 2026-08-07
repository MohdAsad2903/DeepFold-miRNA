import os
os.makedirs("DeepFold_models", exist_ok=True)

with open('d:/Capstone/train_all_models.py', encoding='utf-8') as f:
    code = f.read()

idx = code.find('class DeepFoldCNN_v4')
if idx == -1:
    print("Could not find DeepFoldCNN_v4 in train_all_models.py")
else:
    # Go back to grab imports and dataset definitions if possible, or just add them
    header = """
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

"""
    # Now grab everything AFTER `class DeepFoldCNN_v4:` definition to avoid duplicates
    idx_end = code.find('class DeepFoldCNN_v4')
    tail = code[idx_end:]
    # Actually, we need the df load too
    df_load = """
df = pd.read_csv("DeepFold_Dataset/final_dataset.csv")
print(f"Loaded {len(df)} samples")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
    # Wait, the rest of the code will try to redefine DeepFoldCNN_v4 and SNPDataset_v4.
    # so we can just grab everything from `sgkf = StratifiedGroupKFold(n_splits=5`
    idx_sgkf = code.find('sgkf = StratifiedGroupKFold(n_splits=5')
    if idx_sgkf != -1:
        tail = code[idx_sgkf:]
    else:
        tail = code[idx_end:]

    with open('d:/Capstone/train_final.py', 'w', encoding='utf-8') as f:
        f.write(header + df_load + tail)
    print("train_final.py generated successfully!")
