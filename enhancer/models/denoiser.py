"""
Temporal-Aware Denoising Network
- Edge-preserving spatial denoising using a DnCNN-style architecture
- Temporal averaging across aligned frames when multiple frames available
- Noise level estimation to set adaptive denoise strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DnCNNBlock(nn.Module):
    """DnCNN-style conv-BN-ReLU block."""

    def __init__(self, channels: int = 64, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, 1, padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class NoiseEstimator(nn.Module):
    """
    Estimates a noise level scalar sigma_hat from a patch.
    Used to adaptively scale denoise strength.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),  # positive output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns estimated sigma per batch element (B, 1)."""
        return self.net(x)


class TemporalDenoiserNet(nn.Module):
    """
    Blind denoiser that:
    1. Estimates noise level from input
    2. Applies spatial DnCNN-style denoising
    3. If temporal frames provided, blends with temporally-aligned average

    The noise level estimate is concatenated to features so the network
    is noise-level conditioned without requiring manual sigma input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 15,
    ):
        super().__init__()
        self.noise_estimator = NoiseEstimator(in_channels)

        # Head: image + sigma_map -> features
        self.head = nn.Sequential(
            nn.Conv2d(in_channels + 1, num_features, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Dilated residual blocks for large receptive field
        dilations = [1, 2, 4, 8, 4, 2, 1] * (num_layers // 7 + 1)
        self.body = nn.Sequential(
            *[DnCNNBlock(num_features, d) for d in dilations[:num_layers]]
        )

        # Output residual
        self.tail = nn.Conv2d(num_features, in_channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        aligned_frames: Optional[torch.Tensor] = None,
        temporal_weight: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: noisy input (B, C, H, W)
            aligned_frames: temporally-aligned neighboring frames (B, T, C, H, W) or None
            temporal_weight: blending weight for temporal average (0=spatial only)
        Returns:
            denoised: (B, C, H, W)
            noise_level: estimated sigma (B, 1)
        """
        B, C, H, W = x.shape

        # Estimate noise level
        sigma = self.noise_estimator(x)  # (B, 1)
        sigma_map = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)

        # Spatial denoising
        inp = torch.cat([x, sigma_map], dim=1)
        feat = self.head(inp)
        residual = self.tail(self.body(feat))
        spatial_denoised = x - residual  # learn noise residual

        # Temporal blending
        if aligned_frames is not None:
            # Simple mean across temporally aligned frames
            temporal_avg = aligned_frames.mean(dim=1)  # (B, C, H, W)
            # Weight by how noisy: more noise -> trust temporal more
            sigma_norm = torch.clamp(sigma / 50.0, 0.0, 1.0).view(B, 1, 1, 1)
            tw = temporal_weight * sigma_norm
            denoised = (1 - tw) * spatial_denoised + tw * temporal_avg
        else:
            denoised = spatial_denoised

        return denoised, sigma
