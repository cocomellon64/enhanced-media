"""
Deblurring Network
- Blind deblurring using a U-Net style encoder-decoder
- PSF confidence estimation: if PSF reliability is low, strength is reduced
- Heavy regularization in forensics mode to prevent hallucination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeblurNet(nn.Module):
    """
    Encoder-decoder deblurring network with PSF confidence head.

    The PSF confidence (scalar per-region) gates how much deblurring is applied.
    If confidence < threshold, the input is passed through with minimal change,
    preventing hallucination in uncertain regions.
    """

    def __init__(self, in_channels: int = 3, base_features: int = 32):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)           # H x W
        self.enc2 = ConvBlock(f, f * 2, stride=2)       # H/2
        self.enc3 = ConvBlock(f * 2, f * 4, stride=2)   # H/4
        self.enc4 = ConvBlock(f * 4, f * 8, stride=2)   # H/8

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # Output: residual (blurriness to remove)
        self.out_conv = nn.Conv2d(f, in_channels, 1)

        # PSF confidence head (single scalar map per pixel)
        self.psf_confidence = nn.Sequential(
            nn.Conv2d(f * 8, f * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),  # coarse map
            nn.Conv2d(f * 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        strength: float = 1.0,
        forensics_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: blurry input (B, C, H, W)
            strength: 0.0-1.0 scalar deblur strength multiplier
            forensics_mode: if True, apply heavier regularization and reduce strength
        Returns:
            deblurred: (B, C, H, W)
            psf_confidence: coarse confidence map (B, 1, 8, 8) in [0,1]
        """
        if forensics_mode:
            strength = min(strength, 0.4)  # conservative cap in forensics

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # PSF confidence from bottleneck
        psf_conf = self.psf_confidence(b)  # (B, 1, 8, 8)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = self.out_conv(d1)  # predicted sharpness residual

        # Gate strength by upsampled PSF confidence
        conf_up = F.interpolate(psf_conf, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gated_strength = strength * conf_up

        deblurred = x + gated_strength * residual
        return deblurred, psf_conf
