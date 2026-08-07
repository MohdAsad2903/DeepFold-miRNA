import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import os, random, math

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
NPY_DIR  = "DeepFold_Dataset/processed_maps/npy"
CSV_PATH = "DeepFold_Dataset/final_dataset.csv"
CKPT_DIR = "DeepFold_models"
DEVICE   = torch.device("cpu") # CPU for stability on this env
os.makedirs(CKPT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

class SNPDataset_v4(Dataset):
    def __init__(self, records, augment=False):
        self.records = records.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row    = self.records.iloc[idx]
        tensor = np.load(os.path.join(NPY_DIR, f"{row['Sample_ID']}.npy"))
        tensor = torch.tensor(tensor, dtype=torch.float32).permute(2, 0, 1)
        if self.augment:
            if random.random() > 0.5: tensor = torch.flip(tensor, dims=[1])
            if random.random() > 0.5: tensor = torch.flip(tensor, dims=[2])
        label_idx = torch.tensor(row["Label"], dtype=torch.long)
        return tensor, label_idx, row["miRNA_ID"]

class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(nn.Linear(channels, channels // r, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // r, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * scale

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.10):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.se = SEBlock(out_ch)
        self.project = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout2d(dropout_p)
    def forward(self, x): return self.drop(self.se(self.conv(x)) + self.project(x))

class DeepFoldCNN_v4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.main_path = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.ctx_path = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.input_se = SEBlock(48)
        self.input_drop = nn.Dropout2d(0.10)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ResidualBlock(48, 64, 0.15)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ResidualBlock(64, 128, 0.20)
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = ResidualBlock(128, 256, 0.10)
        self.pool4 = nn.MaxPool2d(2)
        self.pool_final = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, num_classes))
    def forward(self, x):
        x_main = self.main_path(x[:, 2:4, :, :])
        x_ctx = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([x_main, x_ctx], dim=1)
        x = self.pool1(self.input_drop(self.input_se(x)))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return self.classifier(self.flatten(self.pool_final(x)))

def train_one_fold():
    # Only 1 fold for reduced scope as per plan
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df, df["Label"], df["miRNA_ID"]))
    
    print(f"Training on {len(train_idx)} samples, Testing on {len(test_idx)}...")
    
    train_ds = SNPDataset_v4(df.iloc[train_idx], augment=True)
    test_ds  = SNPDataset_v4(df.iloc[test_idx],  augment=False)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)
    
    model = DeepFoldCNN_v4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Just a few epochs for demonstration as per CPU constraints
    EPOCHS = 10
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct = 0, 0
        for tensors, labels, _ in train_dl:
            tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_acc = correct / len(train_ds)
        print(f"Epoch {epoch}: Loss {total_loss/len(train_ds):.4f}, Acc {train_acc:.4f}")

    # Save
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "cnn_v4_fold1.pt"))
    print("Model saved: cnn_v4_fold1.pt")

if __name__ == "__main__":
    # Check if data is ready
    present = len([f for f in os.listdir(NPY_DIR) if f.endswith('.npy')])
    print(f"NPY files available: {present}/{len(df)}")
    if present > 100: # Proceed if we have enough for 1 fold
        train_one_fold()
    else:
        print("Waiting for more data...")
