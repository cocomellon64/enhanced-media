"""
Postprocessing stage
- Conservative color and exposure correction
- Edge-aware unsharp masking (per-mode strength)
- Artifact suppression (halo and ringing detection)
- Final clamp and uint8 conversion
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple


class Postprocessor:
    """
    Applies conservative post-processing after SR and deblur.
    All operations are auditable and logged.
    """

    # Mode-based sharpen strength limits
    SHARPEN_LIMITS = {
        "fast": 0.3,
        "balanced": 0.5,
        "quality": 0.7,
        "conservative": 0.2,
        "aggressive": 1.0,
        "forensics": 0.0,  # no sharpening in forensics
    }

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.max_sharpen = self.SHARPEN_LIMITS.get(mode, 0.5)

    def process(
        self,
        frame_rgb: np.ndarray,
        sharpen_amount: float = 0.5,
        denoise_final: bool = True,
        log: dict = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Args:
            frame_rgb: float32 [0,1] RGB array (H, W, 3)
            sharpen_amount: user-requested sharpen, clamped by mode limit
            denoise_final: apply final light denoise pass
            log: existing log dict to append to
        Returns:
            output_bgr: uint8 BGR for OpenCV writing
            log: updated processing log
        """
        if log is None:
            log = {}

        effective_sharpen = min(sharpen_amount, self.max_sharpen)
        log["postprocess_mode"] = self.mode
        log["sharpen_amount_requested"] = sharpen_amount
        log["sharpen_amount_applied"] = effective_sharpen

        # Color/exposure correction (neutral, conservative)
        frame_rgb, color_ops = self._correct_color(frame_rgb)
        log["color_correction"] = color_ops

        # Edge-aware sharpening
        if effective_sharpen > 0.01 and self.mode != "forensics":
            frame_rgb = self._edge_aware_sharpen(frame_rgb, effective_sharpen)
            log["sharpening_applied"] = True
        else:
            log["sharpening_applied"] = False

        # Final light denoise
        if denoise_final and self.mode not in ("forensics",):
            frame_rgb = self._final_denoise(frame_rgb)
            log["final_denoise_applied"] = True
        else:
            log["final_denoise_applied"] = False

        # Artifact suppression
        frame_rgb = self._suppress_artifacts(frame_rgb, log)

        # Clamp and convert
        frame_rgb = np.clip(frame_rgb, 0.0, 1.0)
        output_bgr = cv2.cvtColor((frame_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return output_bgr, log

    def _correct_color(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Conservative exposure correction only:
        - Stretch histogram if mean luma is too dark or bright
        - No hue/saturation manipulation
        """
        luma = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        mean_luma = float(luma.mean())
        ops = {"mean_luma_before": round(mean_luma, 4)}

        # Only correct if significantly over/under exposed
        if mean_luma < 0.25:
            scale = 0.35 / (mean_luma + 1e-6)
            scale = np.clip(scale, 1.0, 1.5)  # cap at 1.5x boost
            frame = np.clip(frame * scale, 0.0, 1.0)
            ops["exposure_boost"] = round(float(scale), 3)
        elif mean_luma > 0.85:
            scale = 0.75 / (mean_luma + 1e-6)
            scale = np.clip(scale, 0.7, 1.0)
            frame = np.clip(frame * scale, 0.0, 1.0)
            ops["exposure_reduction"] = round(float(scale), 3)
        else:
            ops["exposure_correction"] = "none"

        ops["mean_luma_after"] = round(float(
            (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).mean()
        ), 4)
        return frame, ops

    def _edge_aware_sharpen(self, frame: np.ndarray, amount: float) -> np.ndarray:
        """
        Edge-aware unsharp mask:
        1. Detect edges with Canny
        2. Only apply sharpening in edge regions (avoids noise amplification in flat areas)
        """
        img_u8 = (frame * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=1.5)
        sharp = cv2.addWeighted(img_u8, 1.0 + amount, blurred, -amount, 0)

        # Build edge mask
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_mask = cv2.dilate(edges, kernel).astype(np.float32) / 255.0
        edge_mask_3 = edge_mask[:, :, np.newaxis]

        # Blend: sharpened in edge regions, original elsewhere
        result = (sharp.astype(np.float32) * edge_mask_3 + img_u8.astype(np.float32) * (1 - edge_mask_3))
        return np.clip(result / 255.0, 0.0, 1.0)

    def _final_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Light bilateral filter denoise pass."""
        img_u8 = (frame * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(img_u8, d=5, sigmaColor=15, sigmaSpace=15)
        return denoised.astype(np.float32) / 255.0

    def _suppress_artifacts(self, frame: np.ndarray, log: dict) -> np.ndarray:
        """
        Detects and suppresses halos/ringing:
        - Measures overshoot near strong edges
        - If overshoot > threshold, apply local median filter
        """
        img_u8 = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Measure ringing: high-freq energy away from edges
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
        hf = np.abs(gray - blurred)
        edges = cv2.Canny(img_u8, 40, 120)
        # Dilate edges to exclude near-edge HF (that's desired sharpness)
        non_edge_mask = (cv2.dilate(edges, np.ones((9, 9), np.uint8)) == 0)
        ringing_score = float(hf[non_edge_mask].mean()) if non_edge_mask.any() else 0.0
        log["ringing_score"] = round(ringing_score, 4)

        if ringing_score > 8.0:  # threshold in [0,255] scale
            # Apply light median filter to suppress ringing
            result = cv2.medianBlur(img_u8, 3)
            log["ringing_suppressed"] = True
            return result.astype(np.float32) / 255.0

        log["ringing_suppressed"] = False
        return frame
