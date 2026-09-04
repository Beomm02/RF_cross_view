from __future__ import annotations

import torch
from torch import nn

from rf_multiview_relation.models.decoders import Conv1DDecoder, STFTDecoder
from rf_multiview_relation.models.encoder_ap import APEncoder
from rf_multiview_relation.models.encoder_iq import IQEncoder
from rf_multiview_relation.models.encoder_stft import STFTEncoder


class ViewAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def make_autoencoder(view_name: str, latent_dim: int, stft_shape: tuple[int, int]) -> ViewAutoencoder:
    if view_name == "iq":
        return ViewAutoencoder(IQEncoder(latent_dim=latent_dim), Conv1DDecoder(latent_dim, out_channels=2))
    if view_name == "ap":
        return ViewAutoencoder(APEncoder(latent_dim=latent_dim), Conv1DDecoder(latent_dim, out_channels=2))
    if view_name == "stft":
        return ViewAutoencoder(STFTEncoder(latent_dim=latent_dim), STFTDecoder(latent_dim, target_shape=stft_shape))
    raise ValueError("view_name must be 'iq', 'ap', or 'stft'")
