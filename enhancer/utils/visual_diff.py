"""
Visual diff utilities
- Side-by-side comparison
- Pixel-difference heatmap
"""

import numpy as np
import cv2
from typing import Optional, Tuple


def generate_visual_diff(
    original_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate visual diff between original and enhanced frames.

    Args:
        original_bgr: original BGR uint8 image
        enhanced_bgr: enhanced BGR uint8 image (resized to match if needed)
        output_path: if provided, save composite to this path
    Returns:
        side_by_side: [original | enhanced] panel
        diff_heatmap: pixel-difference heatmap
        composite: [original | enhanced | heatmap]
    """
    # Resize original to match enhanced for comparison
    h, w = enhanced_bgr.shape[:2]
    orig_resized = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # Side-by-side
    divider = np.full((h, 4, 3), 128, dtype=np.uint8)
    side_by_side = np.hstack([orig_resized, divider, enhanced_bgr])

    # Add labels
    cv2.putText(side_by_side, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(side_by_side, "ENHANCED", (w + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Pixel difference heatmap
    diff = cv2.absdiff(orig_resized, enhanced_bgr).astype(np.float32)
    diff_gray = diff.mean(axis=2)
    diff_norm = (diff_gray / diff_gray.max() * 255).clip(0, 255).astype(np.uint8) if diff_gray.max() > 0 else np.zeros_like(diff_gray, dtype=np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Mean absolute difference score
    mad = float(diff_gray.mean())
    cv2.putText(diff_heatmap, f"MAD: {mad:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Composite: original | enhanced | heatmap
    composite = np.hstack([orig_resized, divider, enhanced_bgr, divider, diff_heatmap])

    if output_path:
        cv2.imwrite(output_path, composite)

    return side_by_side, diff_heatmap, composite
