"""
Confidence map visualization utilities.
- Save per-frame confidence maps as float32 numpy files or color-coded images
- Generate hallucination overlay for forensics review
"""

import numpy as np
import cv2
import os
from typing import Optional


# Confidence threshold below which a pixel is "uncertain / hallucinated"
HALLUCINATION_THRESHOLD = 0.4


def save_confidence_map(
    conf: np.ndarray,
    output_path: str,
    as_float32: bool = True,
    colormap: bool = True,
) -> None:
    """
    Save a confidence map.

    Args:
        conf: (H, W) float32 in [0, 1]
        output_path: path without extension (extensions added automatically)
        as_float32: save raw .npy file
        colormap: save color-coded PNG (green=high conf, red=low)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if as_float32:
        np.save(output_path + ".npy", conf.astype(np.float32))

    if colormap:
        # Normalize to uint8
        conf_u8 = (conf * 255).clip(0, 255).astype(np.uint8)
        # Apply colormap: COLORMAP_RdYlGn approximation using JET inverted
        colored = cv2.applyColorMap(255 - conf_u8, cv2.COLORMAP_JET)
        cv2.imwrite(output_path + "_confidence.png", colored)


def overlay_hallucination_mask(
    image_bgr: np.ndarray,
    conf: np.ndarray,
    threshold: float = HALLUCINATION_THRESHOLD,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay semi-transparent red mask on regions flagged as hallucinated.

    Args:
        image_bgr: BGR uint8 image
        conf: (H, W) float32 confidence in [0,1]
        threshold: pixels below this are hallucinated
        alpha: opacity of overlay
    Returns:
        annotated BGR image
    """
    # Resize conf to match image if needed
    h, w = image_bgr.shape[:2]
    if conf.shape != (h, w):
        conf = cv2.resize(conf, (w, h))

    hallucinated = (conf < threshold).astype(np.uint8) * 255

    overlay = image_bgr.copy()
    red_layer = np.zeros_like(image_bgr)
    red_layer[:, :, 2] = hallucinated  # red channel

    mask = (hallucinated > 0)[:, :, np.newaxis]
    overlay = np.where(mask, (alpha * red_layer + (1 - alpha) * image_bgr).astype(np.uint8), image_bgr)

    # Add label
    frac = float((hallucinated > 0).mean())
    label = f"Uncertain: {frac*100:.1f}%"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return overlay
