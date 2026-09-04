from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Conv1DDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, output_length: int = 2048):
        super().__init__()
        self.output_length = int(output_length)
        self.fc = nn.Sequential(
            nn.Linear(int(latent_dim), 128 * 256),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, int(out_channels), kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 128, 256)
        x = self.net(x)
        if x.shape[-1] != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode="linear", align_corners=False)
        return x


class STFTDecoder(nn.Module):
    def __init__(self, latent_dim: int, target_shape: tuple[int, int]):
        super().__init__()
        self.target_shape = (int(target_shape[0]), int(target_shape[1]))
        self.fc = nn.Sequential(
            nn.Linear(int(latent_dim), 128 * 16 * 4),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 128, 16, 4)
        x = F.interpolate(x, size=self.target_shape, mode="bilinear", align_corners=False)
        return self.net(x)
