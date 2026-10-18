"""
Media I/O utilities
- MediaReader: unified reader for images and videos
- MediaWriter: writes enhanced output with correct format
"""

import os
import cv2
import numpy as np
from typing import Generator, Optional, Tuple


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class MediaReader:
    """
    Reads images or video files and yields frames as BGR uint8 arrays.
    """

    def __init__(self, path: str):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            self.media_type = "image"
        elif ext in SUPPORTED_VIDEO_EXTS:
            self.media_type = "video"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._image: Optional[np.ndarray] = None
        self._metadata: dict = {}

    def open(self) -> None:
        if self.media_type == "image":
            self._image = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if self._image is None:
                raise IOError(f"Cannot read image: {self.path}")
            h, w = self._image.shape[:2]
            self._metadata = {"width": w, "height": h, "fps": None, "total_frames": 1}
        else:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video: {self.path}")
            self._metadata = {
                "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self._cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> dict:
        return self._metadata

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yields (frame_index, frame_bgr)."""
        if self.media_type == "image":
            yield 0, self._image.copy()
        else:
            idx = 0
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class MediaWriter:
    """
    Writes enhanced frames to image or video output.
    """

    def __init__(self, path: str, fps: float = 30.0, width: int = 3840, height: int = 2160):
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        ext = os.path.splitext(path)[1].lower()
        self.is_video = ext in SUPPORTED_VIDEO_EXTS
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> None:
        if self.is_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
            if not self._writer.isOpened():
                raise IOError(f"Cannot open video writer: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.is_video:
            self._writer.write(frame_bgr)
        else:
            cv2.imwrite(self.path, frame_bgr)

    def close(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
