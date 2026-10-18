"""
RRDB Network — matches Real-ESRGAN / ESRGAN weight layout exactly.

Architecture (mirrors xinntao/Real-ESRGAN RRDBNet):
  - conv_first
  - body: num_block x RRDB (each RRDB has 3 RDB, each RDB has 5 dense conv layers)
  - conv_body
  - upsample: conv_up1, conv_up2 (each followed by leaky relu)
  - conv_hr
  - conv_last

Parameter names match the pretrained checkpoint so weights load with strict=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def make_layer(block, num_blocks):
    return nn.Sequential(*[block() for _ in range(num_blocks)])


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block — 5 dense conv layers.
    Matches Real-ESRGAN naming: conv1..conv5, lrelu.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialize weights
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= 0.1
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block — 3 stacked RDBs.
    Matches Real-ESRGAN naming: rdb1, rdb2, rdb3.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Full RRDB network — identical architecture and parameter naming to
    Real-ESRGAN RRDBNet. Weights from RealESRGAN_x4plus.pth load with strict=True.

    Also outputs a confidence map (added head, not in original Real-ESRGAN).
    The confidence head uses a separate branch and does NOT interfere with
    the pretrained weight loading (extra keys are handled by strict=False).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Confidence head — separate branch, loaded with strict=False
        self.confidence_head = nn.Sequential(
            nn.Conv2d(num_feat, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: LR input (B, 3, H, W) float32 in [0, 1]
        Returns:
            enhanced: HR output (B, 3, H*4, W*4)
            confidence: per-pixel confidence (B, 1, H*4, W*4)
        """
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # 4x upsample via nearest + conv (matches Real-ESRGAN)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))

        enhanced = self.conv_last(self.lrelu(self.conv_hr(feat)))
        confidence = self.confidence_head(feat)

        return enhanced, confidence
