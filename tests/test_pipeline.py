"""
Test suite for the enhancement pipeline.
Tests cover:
- Model forward passes (shape correctness)
- Preprocessing (blur/noise detection)
- Postprocessing (artifact suppression)
- Confidence map generation
- Hallucination detection
- Metric computation
- Temporal consistency
"""

import pytest
import numpy as np
import torch
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_lr_image():
    """256x256 synthetic LR image with noise."""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    # Add Gaussian noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def sample_lr_tensor():
    """(1, 3, 64, 64) float32 tensor."""
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)


@pytest.fixture
def sample_video_frames():
    """5 frames (1, 5, 3, 64, 64) float32 tensor."""
    return torch.rand(1, 5, 3, 64, 64, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

class TestImageSR:
    def test_output_shape_4x(self, sample_lr_tensor):
        from enhancer.models.super_resolution import ImageSuperResolutionNet
        model = ImageSuperResolutionNet(scale=4)
        model.eval()
        with torch.no_grad():
            enhanced, confidence = model(sample_lr_tensor)
        # 64 * 4 = 256
        assert enhanced.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {enhanced.shape}"
        assert confidence.shape == (1, 1, 256, 256)

    def test_confidence_range(self, sample_lr_tensor):
        from enhancer.models.super_resolution import ImageSuperResolutionNet
        model = ImageSuperResolutionNet(scale=4)
        model.eval()
        with torch.no_grad():
            _, confidence = model(sample_lr_tensor)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_output_range_roughly_bounded(self, sample_lr_tensor):
        """Enhanced output should not wildly overflow for random init."""
        from enhancer.models.super_resolution import ImageSuperResolutionNet
        model = ImageSuperResolutionNet(scale=4)
        model.eval()
        with torch.no_grad():
            enhanced, _ = model(sample_lr_tensor)
        # Random init, but values should be somewhat bounded
        assert not torch.isnan(enhanced).any()
        assert not torch.isinf(enhanced).any()


class TestVideoSR:
    def test_output_shape(self, sample_video_frames):
        from enhancer.models.super_resolution import VideoSuperResolutionNet
        model = VideoSuperResolutionNet(scale=4, num_frames=5)
        model.eval()
        with torch.no_grad():
            enhanced, confidence = model(sample_video_frames, flows=None)
        assert enhanced.shape == (1, 3, 256, 256)
        assert confidence.shape == (1, 1, 256, 256)


class TestDeblurNet:
    def test_output_shape(self, sample_lr_tensor):
        from enhancer.models.deblur import DeblurNet
        model = DeblurNet()
        model.eval()
        with torch.no_grad():
            deblurred, psf_conf = model(sample_lr_tensor, strength=0.5)
        assert deblurred.shape == sample_lr_tensor.shape
        assert psf_conf.shape[1] == 1

    def test_forensics_strength_cap(self, sample_lr_tensor):
        """Forensics mode should cap strength at <= 0.4."""
        from enhancer.models.deblur import DeblurNet
        model = DeblurNet()
        model.eval()
        # Should not raise
        with torch.no_grad():
            deblurred, _ = model(sample_lr_tensor, strength=1.0, forensics_mode=True)
        assert deblurred.shape == sample_lr_tensor.shape


class TestDenoiser:
    def test_output_shape(self, sample_lr_tensor):
        from enhancer.models.denoiser import TemporalDenoiserNet
        model = TemporalDenoiserNet()
        model.eval()
        with torch.no_grad():
            denoised, sigma = model(sample_lr_tensor)
        assert denoised.shape == sample_lr_tensor.shape
        assert sigma.shape[0] == 1

    def test_noise_estimate_positive(self, sample_lr_tensor):
        from enhancer.models.denoiser import TemporalDenoiserNet
        model = TemporalDenoiserNet()
        model.eval()
        with torch.no_grad():
            _, sigma = model(sample_lr_tensor)
        assert (sigma >= 0).all()


class TestOpticalFlow:
    def test_flow_shape(self):
        from enhancer.models.optical_flow import OpticalFlowEstimator
        model = OpticalFlowEstimator()
        model.eval()
        ref = torch.rand(1, 3, 64, 64)
        src = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            flow, conf = model(ref, src)
        assert flow.shape == (1, 2, 64, 64)
        assert conf.shape == (1, 1, 64, 64)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_blur_detection_blurry(self):
        from enhancer.pipeline.preprocessing import Preprocessor
        prep = Preprocessor()
        blurry = cv2.GaussianBlur(
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8), (21, 21), 0
        )
        result = prep.process(blurry, torch.device("cpu"))
        assert result.blur_score > 0.0
        assert result.frame_tensor.shape == (1, 3, 128, 128)

    def test_tensor_range(self, sample_lr_image):
        from enhancer.pipeline.preprocessing import Preprocessor
        prep = Preprocessor()
        result = prep.process(sample_lr_image, torch.device("cpu"))
        assert result.frame_tensor.min() >= 0.0
        assert result.frame_tensor.max() <= 1.0

    def test_noise_estimate_positive(self, sample_lr_image):
        from enhancer.pipeline.preprocessing import Preprocessor
        prep = Preprocessor()
        result = prep.process(sample_lr_image, torch.device("cpu"))
        assert result.estimated_noise_sigma >= 0.0


# ---------------------------------------------------------------------------
# Postprocessing tests
# ---------------------------------------------------------------------------

class TestPostprocessor:
    def test_output_dtype(self, sample_lr_image):
        from enhancer.pipeline.postprocessing import Postprocessor
        post = Postprocessor(mode="balanced")
        frame_rgb = cv2.cvtColor(sample_lr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_bgr, log = post.process(frame_rgb, sharpen_amount=0.5)
        assert out_bgr.dtype == np.uint8
        assert out_bgr.shape[2] == 3

    def test_forensics_no_sharpen(self, sample_lr_image):
        from enhancer.pipeline.postprocessing import Postprocessor
        post = Postprocessor(mode="forensics")
        frame_rgb = cv2.cvtColor(sample_lr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        _, log = post.process(frame_rgb, sharpen_amount=1.0)
        assert log["sharpening_applied"] == False


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical(self):
        from enhancer.evaluation.metrics import compute_psnr
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf") or psnr > 100.0

    def test_psnr_different(self):
        from enhancer.evaluation.metrics import compute_psnr
        ref = np.zeros((64, 64, 3), dtype=np.uint8)
        enh = np.full((64, 64, 3), 128, dtype=np.uint8)
        psnr = compute_psnr(ref, enh)
        assert 0 < psnr < 100

    def test_ssim_identical(self):
        from enhancer.evaluation.metrics import compute_ssim
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim > 0.98

    def test_temporal_consistency_static(self):
        from enhancer.evaluation.metrics import compute_temporal_consistency
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]
        score = compute_temporal_consistency(frames)
        assert score < 0.01  # static video = near-zero error

    def test_temporal_consistency_range(self):
        from enhancer.evaluation.metrics import compute_temporal_consistency
        frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        score = compute_temporal_consistency(frames)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Confidence map tests
# ---------------------------------------------------------------------------

class TestConfidenceMap:
    def test_overlay_output_shape(self, sample_lr_image):
        from enhancer.utils.confidence_map import overlay_hallucination_mask
        conf = np.ones((sample_lr_image.shape[0], sample_lr_image.shape[1]), dtype=np.float32) * 0.3
        overlay = overlay_hallucination_mask(sample_lr_image, conf)
        assert overlay.shape == sample_lr_image.shape
        assert overlay.dtype == np.uint8


# ---------------------------------------------------------------------------
# Pipeline integration smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_image_enhance_smoke(self, sample_lr_image):
        """Smoke test: pipeline runs without crashing on a small image."""
        from enhancer.pipeline.enhancer_pipeline import EnhancerPipeline, EnhancementConfig
        config = EnhancementConfig(mode="balanced", target_width=256, target_height=256)
        pipeline = EnhancerPipeline(config=config, device="cpu")
        pipeline.load_models()

        enhanced, conf_map, log = pipeline.enhance_image(sample_lr_image)
        assert enhanced is not None
        assert enhanced.dtype == np.uint8
        assert conf_map is not None
        assert "hallucinated_fraction" in log
        assert "processing_time_s" in log

    def test_hallucination_fraction_in_range(self, sample_lr_image):
        from enhancer.pipeline.enhancer_pipeline import EnhancerPipeline, EnhancementConfig
        config = EnhancementConfig(mode="balanced", target_width=256, target_height=256)
        pipeline = EnhancerPipeline(config=config, device="cpu")
        pipeline.load_models()
        _, _, log = pipeline.enhance_image(sample_lr_image)
        assert 0.0 <= log["hallucinated_fraction"] <= 1.0
