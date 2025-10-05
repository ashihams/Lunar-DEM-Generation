import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class TransformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


class SFSDeepNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, 64, 7, padding=3)
        self.input_bn = nn.BatchNorm2d(64)
        self.encoder1 = nn.Sequential(ResNetBlock(64), ResNetBlock(64))
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResNetBlock(128), ResNetBlock(128)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResNetBlock(256), ResNetBlock(256)
        )
        self.attention = TransformerAttention(256, num_heads=8)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            ResNetBlock(256), nn.ConvTranspose2d(256, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            ResNetBlock(128), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.slope_x = nn.Conv2d(64, 1, 1)
        self.slope_y = nn.Conv2d(64, 1, 1)
        self.confidence = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x0 = F.relu(self.input_bn(self.input_conv(x)))
        x1 = self.encoder1(x0)
        x1_pool = self.pool1(x1)
        x2 = self.encoder2(x1_pool)
        x2_pool = self.pool2(x2)
        x3 = self.encoder3(x2_pool)
        x3_attn = self.attention(x3)
        d3 = self.decoder3(x3_attn)
        d3_cat = torch.cat([d3, x2], dim=1)
        d2 = self.decoder2(d3_cat)
        d2_cat = torch.cat([d2, x1], dim=1)
        d1 = self.decoder1(d2_cat)
        dzdx = self.slope_x(d1)
        dzdy = self.slope_y(d1)
        conf = self.confidence(d1)
        return dzdx, dzdy, conf


class BayesianUncertaintyEstimator:
    def __init__(self, model, num_samples=20):
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def estimate_uncertainty(self, x):
        self.enable_dropout()
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                dzdx, dzdy, conf = self.model(x)
                predictions.append((dzdx, dzdy, conf))
        dzdx_stack = torch.stack([p[0] for p in predictions])
        dzdy_stack = torch.stack([p[1] for p in predictions])
        dzdx_var = dzdx_stack.var(dim=0)
        dzdy_var = dzdy_stack.var(dim=0)
        uncertainty = torch.sqrt(dzdx_var + dzdy_var)
        return uncertainty


