"""
Optical Flow Estimator
Lightweight coarse-to-fine flow estimation using feature pyramids.
Falls back to zero flow when motion estimation quality is low,
marking those regions in the confidence map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FeaturePyramid(nn.Module):
    """Build a 3-level feature pyramid from an RGB frame."""

    def __init__(self, in_channels: int = 3, feat_channels: int = 32):
        super().__init__()
        self.level0 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 7, 1, 3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.level1 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f0 = self.level0(x)
        f1 = self.level1(f0)
        f2 = self.level2(f1)
        return [f0, f1, f2]


class FlowDecoder(nn.Module):
    """Estimates flow at one pyramid level given cost volume."""

    def __init__(self, feat_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_channels * 2 + 2, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1),  # (u, v)
        )
        # Confidence: how reliable is this flow estimate
        self.confidence = nn.Sequential(
            nn.Conv2d(feat_channels * 2 + 2, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, feat_ref: torch.Tensor, feat_src: torch.Tensor, prev_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([feat_ref, feat_src, prev_flow], dim=1)
        flow_delta = self.net(inp)
        conf = self.confidence(inp)
        return prev_flow + flow_delta, conf


class OpticalFlowEstimator(nn.Module):
    """
    Coarse-to-fine optical flow from source frame to reference frame.
    Returns flow field and per-pixel confidence.
    Low confidence regions -> zero flow applied (no warping), flagged in pipeline.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.pyramid = FeaturePyramid(in_channels, feat_channels=32)
        self.decoder2 = FlowDecoder(feat_channels=128)
        self.decoder1 = FlowDecoder(feat_channels=64)
        self.decoder0 = FlowDecoder(feat_channels=32)

    def forward(
        self, ref: torch.Tensor, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ref: reference frame (B, C, H, W)
            src: source frame (B, C, H, W)
        Returns:
            flow: optical flow from src to ref (B, 2, H, W)
            confidence: per-pixel confidence (B, 1, H, W) in [0,1]
        """
        B, C, H, W = ref.shape
        feats_ref = self.pyramid(ref)
        feats_src = self.pyramid(src)

        # Coarsest level (level 2)
        h2, w2 = feats_ref[2].shape[2], feats_ref[2].shape[3]
        flow = torch.zeros(B, 2, h2, w2, device=ref.device)

        flow, conf2 = self.decoder2(feats_ref[2], feats_src[2], flow)
        flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=False) * 2.0

        flow, conf1 = self.decoder1(feats_ref[1], feats_src[1], flow)
        flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=False) * 2.0

        flow, conf0 = self.decoder0(feats_ref[0], feats_src[0], flow)

        # Upsample to full resolution
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        conf = F.interpolate(conf0, size=(H, W), mode="bilinear", align_corners=False)

        # Zero out flow where confidence is very low (< 0.3)
        low_conf_mask = (conf < 0.3).float()
        flow = flow * (1 - low_conf_mask)

        return flow, conf
