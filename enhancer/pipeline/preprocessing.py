"""
Preprocessing stage
- Detects blur type (motion vs defocus)
- Estimates noise level
- Detects camera shake for stabilization decision
- Normalizes frame to [0, 1] float32 tensor
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PreprocessResult:
    frame_tensor: torch.Tensor          # (1, C, H, W) float32 in [0,1]
    blur_type: str                       # "motion", "defocus", "none", "unknown"
    blur_score: float                    # 0.0 (sharp) to 1.0 (very blurry)
    estimated_noise_sigma: float         # estimated Gaussian noise std (0-255 scale)
    has_camera_shake: bool
    stabilization_needed: bool
    width: int
    height: int
    diagnostics: dict = field(default_factory=dict)


class Preprocessor:
    """
    Analyzes an input frame and prepares it for the enhancement pipeline.
    All analysis is done in numpy/OpenCV for speed.
    """

    def __init__(self, shake_threshold: float = 3.0, blur_threshold: float = 100.0):
        """
        Args:
            shake_threshold: optical flow magnitude threshold for camera shake detection
            blur_threshold: Laplacian variance threshold; below = blurry
        """
        self.shake_threshold = shake_threshold
        self.blur_threshold = blur_threshold
        self._prev_frame_gray: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state (call between clips/videos)."""
        self._prev_frame_gray = None

    def process(self, frame_bgr: np.ndarray, device: torch.device) -> PreprocessResult:
        """
        Args:
            frame_bgr: OpenCV BGR uint8 frame
            device: target torch device
        Returns:
            PreprocessResult with analysis and normalized tensor
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        blur_score, blur_type = self._detect_blur(gray)
        noise_sigma = self._estimate_noise(gray)
        has_shake, flow_magnitude = self._detect_shake(gray)

        # Normalize to [0,1] float32 RGB tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        self._prev_frame_gray = gray.copy()

        return PreprocessResult(
            frame_tensor=tensor,
            blur_type=blur_type,
            blur_score=blur_score,
            estimated_noise_sigma=noise_sigma,
            has_camera_shake=has_shake,
            stabilization_needed=has_shake,
            width=w,
            height=h,
            diagnostics={
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "flow_magnitude_mean": flow_magnitude,
                "noise_sigma": noise_sigma,
            },
        )

    def _detect_blur(self, gray: np.ndarray):
        """
        Uses Laplacian variance to detect blur.
        Low variance = blurry.
        Estimates motion vs defocus via comparing horizontal/vertical gradient balance.
        """
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(np.clip(1.0 - lap_var / self.blur_threshold, 0.0, 1.0))

        if lap_var < self.blur_threshold:
            # Check directional blur: motion blur tends to be directional
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            ratio = float(np.mean(np.abs(sobelx)) / (np.mean(np.abs(sobely)) + 1e-6))
            blur_type = "motion" if ratio > 2.0 or ratio < 0.5 else "defocus"
        else:
            blur_type = "none"

        return blur_score, blur_type

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimates noise sigma using median absolute deviation of high-frequency residual.
        Based on: sigma = median(|wavelet HH band|) / 0.6745
        Approximated via difference of Gaussians.
        """
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        residual = gray.astype(np.float32) - blurred
        sigma = float(np.median(np.abs(residual)) / 0.6745)
        return sigma

    def _detect_shake(self, gray: np.ndarray):
        """
        Detects camera shake using sparse optical flow vs previous frame.
        Returns (has_shake: bool, mean_flow_magnitude: float).
        """
        if self._prev_frame_gray is None:
            return False, 0.0

        # Detect corners for tracking
        corners = cv2.goodFeaturesToTrack(self._prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if corners is None or len(corners) < 10:
            return False, 0.0

        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame_gray, gray, corners, None
        )
        good_old = corners[status.flatten() == 1]
        good_new = new_corners[status.flatten() == 1]

        if len(good_old) < 5:
            return False, 0.0

        flow_vecs = good_new - good_old
        magnitudes = np.linalg.norm(flow_vecs, axis=2).flatten()
        mean_mag = float(np.median(magnitudes))

        has_shake = mean_mag > self.shake_threshold
        return has_shake, mean_mag
