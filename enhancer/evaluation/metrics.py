"""
Quality Metrics
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity (via lpips library)
- Temporal Consistency: mean flow-warp error across frames
"""

import numpy as np
import cv2
from typing import List, Optional
import warnings


def compute_psnr(reference: np.ndarray, enhanced: np.ndarray, max_val: float = 255.0) -> float:
    """
    Compute PSNR between reference (HR ground truth) and enhanced image.

    Args:
        reference: uint8 BGR or float32 image
        enhanced: same format, same size
        max_val: pixel value range max (255 for uint8, 1.0 for float32)
    Returns:
        psnr_db: float (higher is better; typical range 25-45 dB for SR)
    """
    ref = reference.astype(np.float64)
    enh = enhanced.astype(np.float64)

    # Resize if necessary
    if ref.shape != enh.shape:
        enh = cv2.resize(enh.astype(np.float32), (ref.shape[1], ref.shape[0])).astype(np.float64)

    mse = np.mean((ref - enh) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(
    reference: np.ndarray,
    enhanced: np.ndarray,
    channel_axis: int = -1,
) -> float:
    """
    Compute SSIM. Uses scikit-image if available, else a simplified NumPy implementation.

    Args:
        reference: uint8 or float32 image
        enhanced: same format, same size
        channel_axis: axis for color channels (-1 for HWC)
    Returns:
        ssim_score: float in [-1, 1], higher is better
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ref_f = reference.astype(np.float32) / 255.0 if reference.dtype == np.uint8 else reference.astype(np.float32)
        enh_f = enhanced.astype(np.float32) / 255.0 if enhanced.dtype == np.uint8 else enhanced.astype(np.float32)
        if ref_f.shape != enh_f.shape:
            enh_f = cv2.resize(enh_f, (ref_f.shape[1], ref_f.shape[0]))
        return float(ssim_fn(ref_f, enh_f, channel_axis=channel_axis, data_range=1.0))
    except ImportError:
        return _ssim_numpy(reference, enhanced)


def _ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified single-channel SSIM computation.
    Converts to grayscale first.
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if img2.ndim == 3:
        if img2.shape != img1.shape[::-1]:
            img2 = cv2.resize(img2.astype(np.uint8), (img1.shape[1], img1.shape[0]))
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1.astype(np.float32), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float32), (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur((img1 ** 2).astype(np.float32), (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur((img2 ** 2).astype(np.float32), (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur((img1 * img2).astype(np.float32), (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def compute_lpips(
    reference: np.ndarray,
    enhanced: np.ndarray,
    net: str = "alex",
) -> Optional[float]:
    """
    Compute LPIPS (perceptual similarity). Requires the `lpips` package.
    Returns None if lpips is not installed.

    Args:
        reference, enhanced: uint8 BGR images
        net: "alex" (fast) or "vgg" (quality)
    Returns:
        lpips_score: float (lower is more similar, ~0.0 perfect)
    """
    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net=net, verbose=False)

        def bgr_to_lpips_tensor(img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        ref_t = bgr_to_lpips_tensor(reference)
        enh_t = bgr_to_lpips_tensor(
            cv2.resize(enhanced, (reference.shape[1], reference.shape[0]))
            if enhanced.shape[:2] != reference.shape[:2] else enhanced
        )
        with torch.no_grad():
            score = loss_fn(ref_t, enh_t)
        return float(score.item())
    except ImportError:
        warnings.warn("lpips not installed; LPIPS metric skipped. pip install lpips")
        return None


def compute_temporal_consistency(
    frames: List[np.ndarray],
) -> float:
    """
    Compute temporal consistency score as mean flow-warp error across consecutive frames.
    Lower is better (less flicker).

    Args:
        frames: list of BGR uint8 frames in temporal order
    Returns:
        mean_warp_error: float
    """
    if len(frames) < 2:
        return 0.0

    errors = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i in range(1, len(frames)):
        curr = frames[i]
        h, w = curr.shape[:2]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Farneback dense flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray.astype(np.uint8), curr_gray.astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        ).astype(np.float32)

        # Warp previous frame forward
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[:, :, 0]).astype(np.float32)
        map_y = (map_y + flow[:, :, 1]).astype(np.float32)

        prev_bgr = frames[i - 1]
        warped = cv2.remap(
            prev_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        ).astype(np.float32)

        curr_f = curr.astype(np.float32)
        warp_error = float(np.mean(np.abs(curr_f - warped)) / 255.0)
        errors.append(warp_error)

        prev_gray = curr_gray

    return float(np.mean(errors))
