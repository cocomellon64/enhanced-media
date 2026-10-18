"""
Temporal Consistency Smoother
- Enforces temporal consistency on enhanced frames to prevent flicker
- Uses exponential moving average with motion-adaptive blending
- Computes warp error metric for temporal consistency score
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple


class TemporalSmoother:
    """
    Maintains a rolling buffer of recent enhanced frames and blends
    them to suppress flicker and temporal artifacts.

    Motion-adaptive: fast-moving regions get less smoothing to preserve
    sharpness; static regions get more smoothing.
    """

    def __init__(
        self,
        window: int = 3,
        blend_alpha: float = 0.3,
        motion_threshold: float = 5.0,
    ):
        """
        Args:
            window: number of past frames to keep in buffer
            blend_alpha: EMA weight for temporal blending (0=all past, 1=all current)
            motion_threshold: flow magnitude threshold to reduce smoothing
        """
        self.window = window
        self.blend_alpha = blend_alpha
        self.motion_threshold = motion_threshold
        self._buffer: deque = deque(maxlen=window)
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self._buffer.clear()
        self._prev_gray = None

    def smooth(
        self, frame_rgb: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Args:
            frame_rgb: current enhanced frame, float32 [0,1] (H, W, 3)
        Returns:
            smoothed_frame: temporally smoothed frame
            warp_error: flow-warp consistency error (lower = better)
        """
        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute warp error against previous frame
        warp_error = 0.0
        motion_mask = np.ones((h, w), dtype=np.float32)

        if self._prev_gray is not None and len(self._buffer) > 0:
            # Dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

            # Warp previous frame using flow
            prev_frame = self._buffer[-1]
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow[:, :, 0]).astype(np.float32)
            map_y = (map_y + flow[:, :, 1]).astype(np.float32)
            warped_prev = cv2.remap(
                (prev_frame * 255).astype(np.uint8),
                map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            ).astype(np.float32) / 255.0
            warp_error = float(np.mean(np.abs(frame_rgb - warped_prev)))

            # Motion-adaptive mask: reduce blending in fast-moving areas
            motion_mask = np.clip(1.0 - mag / self.motion_threshold, 0.0, 1.0)[:, :, np.newaxis]

        # EMA blend with past frames
        if len(self._buffer) > 0:
            past = self._buffer[-1]
            # In static areas: blend more with past; in motion areas: use current
            smoothed = (
                self.blend_alpha * frame_rgb +
                (1.0 - self.blend_alpha) * past
            ) * motion_mask + frame_rgb * (1.0 - motion_mask)
        else:
            smoothed = frame_rgb.copy()

        self._buffer.append(smoothed.copy())
        self._prev_gray = gray.copy()

        return np.clip(smoothed, 0.0, 1.0), warp_error
