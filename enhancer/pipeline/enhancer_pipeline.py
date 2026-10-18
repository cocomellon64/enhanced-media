"""
Main Enhancement Pipeline
Orchestrates the full 10-stage pipeline:
  1. Preprocess
  2. Stabilize (optional)
  3. Denoise
  4. Motion estimation (optical flow)
  5. Multi-frame SR fusion (or single-image SR fallback)
  6. Deblur
  7. Per-frame refinement
  8. Temporal smoothing
  9. Postprocess
  10. Output + confidence maps + processing log
"""

import time
import hashlib
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict, Any

from enhancer.models.super_resolution import ImageSuperResolutionNet, VideoSuperResolutionNet
from enhancer.models.rrdb_net import RRDBNet
from enhancer.models.deblur import DeblurNet
from enhancer.models.denoiser import TemporalDenoiserNet
from enhancer.models.optical_flow import OpticalFlowEstimator
from enhancer.pipeline.preprocessing import Preprocessor, PreprocessResult
from enhancer.pipeline.postprocessing import Postprocessor
from enhancer.pipeline.temporal_smoother import TemporalSmoother


@dataclass
class EnhancementConfig:
    target_width: int = 3840
    target_height: int = 2160
    mode: str = "balanced"          # fast | balanced | quality | forensics
    aggressive: bool = False
    stabilize: bool = True
    preserve_aspect: bool = True
    sharpen_amount: float = 0.5     # 0.0-1.0
    denoise_strength: float = 0.5   # 0.0-1.0
    temporal_smoothing: float = 0.4
    hallucination_tolerance: float = 0.1  # max fraction before warning
    num_context_frames: int = 5     # frames for multi-frame SR


@dataclass
class FrameResult:
    frame_index: int
    enhanced_bgr: np.ndarray
    confidence_map: np.ndarray      # float32 [0,1] (H, W)
    hallucinated_fraction: float
    warp_error: float
    psf_confidence: float
    noise_sigma: float
    blur_type: str
    processing_time_ms: float
    operations_log: dict = field(default_factory=dict)


@dataclass
class ProcessingLog:
    version: str = "1.0.0"
    mode: str = ""
    target_resolution: list = field(default_factory=list)
    total_frames: int = 0
    fps: float = 0.0
    total_runtime_s: float = 0.0
    gpu_used: bool = False
    frame_logs: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    model_checksum: str = ""
    hallucinated_frames: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# Hallucination threshold: confidence below this is "uncertain"
HALLUCINATION_CONFIDENCE_THRESHOLD = 0.4


class EnhancerPipeline:
    """
    Full video/image enhancement pipeline.

    Usage:
        pipeline = EnhancerPipeline(config)
        pipeline.load_models()
        result = pipeline.enhance_image(image_bgr)
        # or
        pipeline.enhance_video(input_path, output_path, callback=progress_fn)
    """

    MODEL_VERSION = "1.0.0"

    def __init__(self, config: Optional[EnhancementConfig] = None, device: Optional[str] = None):
        self.config = config or EnhancementConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.gpu_used = self.device.type == "cuda"

        # Models (loaded lazily)
        self._image_sr: Optional[ImageSuperResolutionNet] = None
        self._video_sr: Optional[VideoSuperResolutionNet] = None
        self._deblur: Optional[DeblurNet] = None
        self._denoiser: Optional[TemporalDenoiserNet] = None
        self._flow_estimator: Optional[OpticalFlowEstimator] = None

        # Pipeline stages
        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor(mode=self.config.mode)
        self.temporal_smoother = TemporalSmoother(
            blend_alpha=self.config.temporal_smoothing
        )

        # Frame buffer for multi-frame SR
        self._frame_buffer: deque = deque(maxlen=self.config.num_context_frames)
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_models(self, checkpoint_dir: Optional[str] = None) -> None:
        """
        Load or initialize all neural network models.
        If checkpoint_dir is provided, loads pretrained weights.
        Otherwise initializes with random weights (for development/testing).
        """
        scale = self._compute_scale()

        self._image_sr = RRDBNet(scale=scale).to(self.device)
        self._video_sr = VideoSuperResolutionNet(
            scale=scale, num_frames=self.config.num_context_frames
        ).to(self.device)
        self._deblur = DeblurNet().to(self.device)
        self._denoiser = TemporalDenoiserNet().to(self.device)
        self._flow_estimator = OpticalFlowEstimator().to(self.device)

        if checkpoint_dir:
            self._load_checkpoints(checkpoint_dir)

        # Set mode
        if self.config.mode == "fast":
            for m in [self._image_sr, self._video_sr, self._deblur, self._denoiser, self._flow_estimator]:
                m.eval()
        else:
            for m in [self._image_sr, self._video_sr, self._deblur, self._denoiser, self._flow_estimator]:
                m.eval()

        self._models_loaded = True
        print(f"[Pipeline] Models loaded on {self.device}. Scale: {scale}x")

    def _load_checkpoints(self, checkpoint_dir: str) -> None:
        """Load model weights from checkpoint directory."""
        import os
        checkpoints = {
            "image_sr": self._image_sr,
            "video_sr": self._video_sr,
            "deblur": self._deblur,
            "denoiser": self._denoiser,
            "flow": self._flow_estimator,
        }
        for name, model in checkpoints.items():
            path = os.path.join(checkpoint_dir, f"{name}.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device, weights_only=False)
                # Unwrap Real-ESRGAN / ESRGAN style checkpoints
                if isinstance(state, dict) and "params_ema" in state:
                    state = state["params_ema"]
                elif isinstance(state, dict) and "params" in state:
                    state = state["params"]
                missing, unexpected = model.load_state_dict(state, strict=False)
                loaded = len(model.state_dict()) - len(missing)
                print(f"[Pipeline] Loaded checkpoint: {name} "
                      f"({loaded} matched, {len(missing)} missing, {len(unexpected)} unexpected)")
            else:
                print(f"[Pipeline] No checkpoint found for {name}, using random init")

    def _compute_scale(self) -> int:
        """Determine upscaling factor. Always 4x (for HD->4K) or 2x (for 4K source)."""
        # Default: assume input may be up to 1080p -> 4K = 4x
        # Actual scale applied may be less if input is already large
        return 4

    # ------------------------------------------------------------------
    # Single-image enhancement
    # ------------------------------------------------------------------

    def enhance_image(
        self, image_bgr: np.ndarray, reference_bgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Enhance a single image.

        Args:
            image_bgr: input BGR uint8 image
            reference_bgr: optional high-quality reference for metric computation
        Returns:
            enhanced_bgr: output enhanced BGR uint8
            confidence_map: float32 (H, W) per-pixel confidence
            log: processing log dict
        """
        if not self._models_loaded:
            self.load_models()

        t0 = time.time()
        log = {
            "pipeline_version": self.MODEL_VERSION,
            "mode": self.config.mode,
            "input_resolution": f"{image_bgr.shape[1]}x{image_bgr.shape[0]}",
        }

        # Stage 1: Preprocess
        prep = self.preprocessor.process(image_bgr, self.device)
        log.update({
            "blur_type": prep.blur_type,
            "blur_score": round(prep.blur_score, 3),
            "noise_sigma": round(prep.estimated_noise_sigma, 2),
            "diagnostics": prep.diagnostics,
        })

        frame_t = prep.frame_tensor  # (1, 3, H, W)

        # Stage 3: Denoise
        with torch.no_grad():
            denoised_t, noise_est = self._denoiser(frame_t)
        log["noise_estimated_sigma"] = round(float(noise_est.mean()), 2)

        # Stage 5: Single-image SR
        with torch.no_grad():
            enhanced_t, confidence_t = self._image_sr(denoised_t)
        enhanced_t = self._apply_aspect_crop(enhanced_t, image_bgr.shape)

        # Stage 6: Deblur (only if blur detected)
        if prep.blur_score > 0.2 and self.config.mode != "forensics":
            with torch.no_grad():
                strength = 0.5 if not self.config.aggressive else 0.8
                enhanced_t, psf_conf = self._deblur(enhanced_t, strength=strength)
            log["deblur_applied"] = True
            log["deblur_strength"] = strength
            log["psf_confidence_mean"] = round(float(psf_conf.mean()), 3)
        elif self.config.mode == "forensics":
            with torch.no_grad():
                enhanced_t, psf_conf = self._deblur(enhanced_t, strength=0.3, forensics_mode=True)
            log["deblur_applied"] = True
            log["forensics_mode"] = True
        else:
            log["deblur_applied"] = False

        # Confidence and hallucination analysis
        conf_np = confidence_t.squeeze().cpu().numpy()  # (H, W)
        conf_np = cv2.resize(conf_np, (
            min(self.config.target_width, enhanced_t.shape[3]),
            min(self.config.target_height, enhanced_t.shape[2])
        ))
        hallu_frac = float((conf_np < HALLUCINATION_CONFIDENCE_THRESHOLD).mean())
        log["hallucinated_fraction"] = round(hallu_frac, 4)
        if hallu_frac > self.config.hallucination_tolerance:
            log["hallucination_warning"] = (
                f"{hallu_frac*100:.1f}% of pixels flagged as uncertain (threshold: "
                f"{self.config.hallucination_tolerance*100:.0f}%)"
            )

        # Convert enhanced tensor to numpy
        enhanced_np = self._tensor_to_numpy(enhanced_t)

        # Stage 9: Postprocess
        enhanced_bgr, post_log = self.postprocessor.process(
            enhanced_np,
            sharpen_amount=self.config.sharpen_amount,
            log=log,
        )
        log.update(post_log)

        elapsed = time.time() - t0
        log["processing_time_s"] = round(elapsed, 3)
        log["output_resolution"] = f"{enhanced_bgr.shape[1]}x{enhanced_bgr.shape[0]}"

        return enhanced_bgr, conf_np, log

    # ------------------------------------------------------------------
    # Video enhancement
    # ------------------------------------------------------------------

    def enhance_video(
        self,
        input_path: str,
        output_path: str,
        callback=None,
    ) -> ProcessingLog:
        """
        Enhance a video file end-to-end.

        Args:
            input_path: path to input video
            output_path: path for enhanced output video
            callback: optional fn(frame_idx, total_frames, fps_estimate) for progress
        Returns:
            ProcessingLog
        """
        if not self._models_loaded:
            self.load_models()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute output resolution
        out_w, out_h = self._compute_output_resolution(src_w, src_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (out_w, out_h))

        plog = ProcessingLog(
            mode=self.config.mode,
            target_resolution=[out_w, out_h],
            total_frames=total_frames,
            fps=src_fps,
            gpu_used=self.gpu_used,
        )

        self.preprocessor.reset()
        self.temporal_smoother.reset()
        self._frame_buffer.clear()

        t_start = time.time()
        frame_idx = 0
        warp_errors = []

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            t_frame = time.time()
            frame_result = self._enhance_video_frame(frame_bgr, frame_idx)
            frame_ms = (time.time() - t_frame) * 1000

            writer.write(frame_result.enhanced_bgr)
            warp_errors.append(frame_result.warp_error)

            plog.frame_logs.append({
                "frame": frame_idx,
                "blur_type": frame_result.blur_type,
                "noise_sigma": round(frame_result.noise_sigma, 2),
                "hallucinated_fraction": round(frame_result.hallucinated_fraction, 4),
                "warp_error": round(frame_result.warp_error, 5),
                "processing_ms": round(frame_ms, 1),
            })

            if frame_result.hallucinated_fraction > self.config.hallucination_tolerance:
                plog.hallucinated_frames.append(frame_idx)

            frame_idx += 1
            if callback:
                elapsed = time.time() - t_start
                fps_est = frame_idx / elapsed if elapsed > 0 else 0
                callback(frame_idx, total_frames, fps_est)

        cap.release()
        writer.release()

        elapsed_total = time.time() - t_start
        plog.total_runtime_s = round(elapsed_total, 2)
        plog.metrics = {
            "temporal_consistency_score": round(float(np.mean(warp_errors)), 5),
            "mean_processing_ms_per_frame": round(elapsed_total * 1000 / max(frame_idx, 1), 1),
            "effective_fps": round(frame_idx / elapsed_total, 2) if elapsed_total > 0 else 0,
        }
        plog.model_checksum = self._compute_model_checksum()

        return plog

    def _enhance_video_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameResult:
        """Process a single video frame using the full pipeline."""
        t0 = time.time()
        ops_log = {"frame_index": frame_idx}

        # Stage 1: Preprocess
        prep = self.preprocessor.process(frame_bgr, self.device)

        # Stage 3: Denoise (temporal if we have context frames)
        with torch.no_grad():
            if len(self._frame_buffer) > 0:
                aligned_stack = torch.stack([f for f in self._frame_buffer], dim=1)
                denoised_t, noise_est = self._denoiser(
                    prep.frame_tensor, aligned_frames=aligned_stack
                )
            else:
                denoised_t, noise_est = self._denoiser(prep.frame_tensor)

        # Stage 4: Optical flow + multi-frame SR
        if len(self._frame_buffer) >= 2:
            frames_for_sr = list(self._frame_buffer) + [denoised_t]
            # Pad or trim to num_context_frames
            n = self.config.num_context_frames
            while len(frames_for_sr) < n:
                frames_for_sr.insert(0, frames_for_sr[0])
            frames_for_sr = frames_for_sr[-n:]

            # Compute flows for all frames to center
            ref = denoised_t
            flows = []
            for ft in frames_for_sr:
                if torch.equal(ft, ref):
                    flows.append(torch.zeros(1, 2, ref.shape[2], ref.shape[3], device=self.device))
                else:
                    with torch.no_grad():
                        flow, _ = self._flow_estimator(ref, ft)
                    flows.append(flow.unsqueeze(1))

            frames_tensor = torch.stack(frames_for_sr, dim=1)  # (1, T, C, H, W)
            flows_tensor = torch.cat(flows, dim=1)              # (1, T, 2, H, W)

            with torch.no_grad():
                enhanced_t, confidence_t = self._video_sr(frames_tensor, flows_tensor)
            ops_log["sr_mode"] = "multi_frame"
            ops_log["frames_used"] = len(frames_for_sr)
        else:
            # Fallback: single-image SR
            with torch.no_grad():
                enhanced_t, confidence_t = self._image_sr(denoised_t)
            ops_log["sr_mode"] = "single_image"

        # Stage 6: Deblur
        psf_conf_mean = 1.0
        if prep.blur_score > 0.2:
            forensics = self.config.mode == "forensics"
            strength = 0.3 if forensics else (0.7 if self.config.aggressive else 0.4)
            with torch.no_grad():
                enhanced_t, psf_conf = self._deblur(enhanced_t, strength=strength, forensics_mode=forensics)
            psf_conf_mean = float(psf_conf.mean())
            ops_log["deblur_applied"] = True

        # Update frame buffer
        self._frame_buffer.append(denoised_t)

        # Confidence analysis
        conf_np = confidence_t.squeeze().cpu().numpy()
        enhanced_np = self._tensor_to_numpy(enhanced_t)

        # Resize conf_np to match enhanced output
        h_out, w_out = enhanced_np.shape[:2]
        if conf_np.shape != (h_out, w_out):
            conf_np = cv2.resize(conf_np, (w_out, h_out))
        hallu_frac = float((conf_np < HALLUCINATION_CONFIDENCE_THRESHOLD).mean())

        # Stage 8: Temporal smoothing
        smoothed_np, warp_error = self.temporal_smoother.smooth(enhanced_np)

        # Stage 9: Postprocess
        enhanced_bgr, post_log = self.postprocessor.process(
            smoothed_np,
            sharpen_amount=self.config.sharpen_amount,
            log=ops_log,
        )

        elapsed_ms = (time.time() - t0) * 1000
        return FrameResult(
            frame_index=frame_idx,
            enhanced_bgr=enhanced_bgr,
            confidence_map=conf_np,
            hallucinated_fraction=hallu_frac,
            warp_error=warp_error,
            psf_confidence=psf_conf_mean,
            noise_sigma=float(noise_est.mean()),
            blur_type=prep.blur_type,
            processing_time_ms=elapsed_ms,
            operations_log=ops_log,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """(1, C, H, W) float32 tensor -> (H, W, C) float32 numpy [0,1]."""
        return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    def _apply_aspect_crop(self, tensor: torch.Tensor, src_shape) -> torch.Tensor:
        """
        If preserve_aspect=True, resize to fit target while keeping aspect ratio.
        """
        src_h, src_w = src_shape[:2]
        target_h, target_w = self.config.target_height, self.config.target_width

        if self.config.preserve_aspect:
            ratio = min(target_w / (src_w * self._compute_scale()),
                        target_h / (src_h * self._compute_scale()))
            new_w = int(src_w * self._compute_scale() * ratio)
            new_h = int(src_h * self._compute_scale() * ratio)
        else:
            new_w, new_h = target_w, target_h

        return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def _compute_output_resolution(self, src_w: int, src_h: int) -> Tuple[int, int]:
        """Compute output resolution respecting aspect ratio."""
        target_w, target_h = self.config.target_width, self.config.target_height
        if self.config.preserve_aspect:
            ratio = min(target_w / src_w, target_h / src_h)
            return int(src_w * ratio), int(src_h * ratio)
        return target_w, target_h

    def _compute_model_checksum(self) -> str:
        """SHA256 of model parameter shapes as a simple version fingerprint."""
        shapes = []
        for m in [self._image_sr, self._video_sr, self._deblur, self._denoiser]:
            if m is not None:
                for p in m.parameters():
                    shapes.append(str(p.shape))
        checksum = hashlib.sha256("|".join(shapes).encode()).hexdigest()[:16]
        return checksum

    def generate_manifest(self, output_file: str, log: ProcessingLog) -> str:
        """
        Write a processing manifest JSON alongside output files.
        Includes model version, parameters, and checksum for audit.
        """
        manifest = {
            "manifest_version": "1.0",
            "model_version": self.MODEL_VERSION,
            "model_checksum": log.model_checksum,
            "mode": log.mode,
            "target_resolution": log.target_resolution,
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {},
            "metrics": log.metrics,
            "total_frames": log.total_frames,
            "total_runtime_s": log.total_runtime_s,
            "gpu_used": log.gpu_used,
            "hallucinated_frames_count": len(log.hallucinated_frames),
            "hallucinated_frame_indices": log.hallucinated_frames,
            "warnings": log.warnings,
        }
        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=2)
        return output_file
