"""
Extended metric tests with known synthetic cases.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_psnr_known_value():
    """PSNR for a fixed MSE should match formula."""
    from enhancer.evaluation.metrics import compute_psnr
    ref = np.zeros((4, 4, 3), dtype=np.float32)
    enh = np.full((4, 4, 3), 10.0 / 255.0, dtype=np.float32)
    mse = (10.0 ** 2)
    expected_psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    computed = compute_psnr(
        (ref * 255).astype(np.uint8),
        (enh * 255).astype(np.uint8),
    )
    assert abs(computed - expected_psnr) < 1.0


def test_ssim_range():
    from enhancer.evaluation.metrics import compute_ssim
    img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    ssim = compute_ssim(img1, img2)
    assert -1.0 <= ssim <= 1.0


def test_temporal_consistency_single_frame():
    from enhancer.evaluation.metrics import compute_temporal_consistency
    frames = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)]
    score = compute_temporal_consistency(frames)
    assert score == 0.0


def test_evaluator_report_no_reference():
    from enhancer.evaluation.evaluator import Evaluator
    originals = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    enhanced = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    conf_maps = [np.random.rand(64, 64).astype(np.float32) for _ in range(3)]

    evaluator = Evaluator()
    report = evaluator.evaluate(originals, enhanced, confidence_maps=conf_maps)

    assert report.total_frames == 3
    assert report.mean_psnr is None  # no reference
    assert 0.0 <= report.temporal_consistency <= 1.0
    assert 0.0 <= report.mean_hallucinated_fraction <= 1.0
