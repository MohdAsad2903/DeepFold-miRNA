import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.main_path = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.ctx_path = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.input_attn = CoordAttention(48, 48)
        self.input_drop = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d(2) 
        self.block2 = ResidualBlock(48, 64, 0.15)
        self.pool2 = nn.MaxPool2d(2) 
        self.block3 = ResidualBlock(64, 128, 0.20)
        self.pool3 = nn.MaxPool2d(2) 
        self.block4 = ResidualBlock(128, 256, 0.10)
        self.pool4 = nn.MaxPool2d(2) 
        self.pool_final = MultiScalePool()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        xm = self.main_path(x[:, 2:4, :, :])
        xc = self.ctx_path(x[:, 0:2, :, :])
        x = torch.cat([xm, xc], dim=1)
        x = self.pool1(self.input_drop(self.input_attn(x)))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return self.classifier(self.flatten(self.pool_final(x)))

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
