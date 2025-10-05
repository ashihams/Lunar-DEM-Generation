import torch
import torch.nn as nn
import numpy as np

from .model import ResNetBlock


class DiffusionDenoisingModel(nn.Module):
    def __init__(self, channels=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.time_embed = nn.Sequential(nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.down1 = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU())
        self.down2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.GroupNorm(8, channels * 2), nn.SiLU()
        )
        self.mid = nn.Sequential(ResNetBlock(channels * 2), ResNetBlock(channels * 2))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(channels * 2, channels, 2, stride=2), nn.GroupNorm(8, channels), nn.SiLU())
        self.up1 = nn.Conv2d(channels * 2, 2, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t.view(-1, 1))
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1)
        d1 = self.down1(x)
        d1 = d1 + t_emb[:, :d1.shape[1], :, :]
        d2 = self.down2(d1)
        m = self.mid(d2)
        u2 = self.up2(m)
        u1 = torch.cat([u2, d1], dim=1)
        out = self.up1(u1)
        return out


class SubPixelRefinement:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DiffusionDenoisingModel().to(self.device)
        self.model.eval()

    def refine_slopes(self, dzdx: np.ndarray, dzdy: np.ndarray, num_steps=50):
        slopes = np.stack([dzdx, dzdy], axis=0)
        slopes_tensor = torch.from_numpy(slopes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for t in range(num_steps):
                t_tensor = torch.tensor([t / num_steps], device=self.device)
                noise_pred = self.model(slopes_tensor, t_tensor)
                slopes_tensor = slopes_tensor - 0.01 * noise_pred
        refined = slopes_tensor.squeeze().cpu().numpy()
        return refined[0], refined[1]


