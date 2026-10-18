"""
Evaluator
Runs the full metric suite on a set of (original, enhanced) pairs
and generates a JSON report.
"""

import os
import json
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict

from .metrics import compute_psnr, compute_ssim, compute_lpips, compute_temporal_consistency


@dataclass
class FrameMetrics:
    frame_index: int
    psnr: Optional[float]       # None if no reference
    ssim: Optional[float]
    lpips: Optional[float]
    hallucinated_fraction: float


@dataclass
class EvaluationReport:
    dataset: str
    total_frames: int
    mean_psnr: Optional[float]
    mean_ssim: Optional[float]
    mean_lpips: Optional[float]
    temporal_consistency: float
    mean_hallucinated_fraction: float
    no_hallucination_indicator: float  # fraction of frames with hallu < threshold
    per_frame: List[dict]


class Evaluator:
    """
    Evaluates enhancement quality against reference images/video.

    Usage:
        evaluator = Evaluator(hallucination_threshold=0.05)
        report = evaluator.evaluate(
            originals=[bgr1, bgr2, ...],
            enhanced=[enh1, enh2, ...],
            confidence_maps=[conf1, conf2, ...],
            references=[ref1, ref2, ...],  # optional HR references
        )
        evaluator.save_report(report, "report.json")
    """

    def __init__(self, hallucination_threshold: float = 0.05):
        self.hallucination_threshold = hallucination_threshold

    def evaluate(
        self,
        originals: List[np.ndarray],
        enhanced: List[np.ndarray],
        confidence_maps: Optional[List[np.ndarray]] = None,
        references: Optional[List[np.ndarray]] = None,
        dataset_name: str = "unnamed",
    ) -> EvaluationReport:
        """
        Args:
            originals: list of LR BGR input frames
            enhanced: list of enhanced BGR output frames
            confidence_maps: list of float32 (H,W) confidence arrays
            references: list of HR ground truth BGR frames (optional, for supervised metrics)
            dataset_name: label for the report
        Returns:
            EvaluationReport
        """
        n = len(enhanced)
        per_frame_metrics = []
        psnrs, ssims, lpips_scores = [], [], []

        for i, enh in enumerate(enhanced):
            ref = references[i] if (references and i < len(references)) else None

            psnr = compute_psnr(ref, enh) if ref is not None else None
            ssim = compute_ssim(ref, enh) if ref is not None else None
            lpips_val = compute_lpips(ref, enh) if ref is not None else None

            conf = confidence_maps[i] if (confidence_maps and i < len(confidence_maps)) else None
            hallu_frac = float((conf < 0.4).mean()) if conf is not None else 0.0

            per_frame_metrics.append(FrameMetrics(
                frame_index=i,
                psnr=round(psnr, 3) if psnr else None,
                ssim=round(ssim, 4) if ssim else None,
                lpips=round(lpips_val, 4) if lpips_val else None,
                hallucinated_fraction=round(hallu_frac, 4),
            ))

            if psnr:
                psnrs.append(psnr)
            if ssim:
                ssims.append(ssim)
            if lpips_val:
                lpips_scores.append(lpips_val)

        # Temporal consistency
        tc = compute_temporal_consistency(enhanced) if len(enhanced) > 1 else 0.0

        mean_hallu = float(np.mean([m.hallucinated_fraction for m in per_frame_metrics]))
        no_hallu_indicator = float(
            np.mean([m.hallucinated_fraction < self.hallucination_threshold for m in per_frame_metrics])
        )

        return EvaluationReport(
            dataset=dataset_name,
            total_frames=n,
            mean_psnr=round(float(np.mean(psnrs)), 3) if psnrs else None,
            mean_ssim=round(float(np.mean(ssims)), 4) if ssims else None,
            mean_lpips=round(float(np.mean(lpips_scores)), 4) if lpips_scores else None,
            temporal_consistency=round(tc, 5),
            mean_hallucinated_fraction=round(mean_hallu, 4),
            no_hallucination_indicator=round(no_hallu_indicator, 4),
            per_frame=[asdict(m) for m in per_frame_metrics],
        )

    def save_report(self, report: EvaluationReport, output_path: str) -> None:
        """Save evaluation report as JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[Evaluator] Report saved to {output_path}")
        self._print_summary(report)

    def _print_summary(self, report: EvaluationReport) -> None:
        print(f"\n=== Evaluation Report: {report.dataset} ===")
        print(f"  Frames evaluated   : {report.total_frames}")
        print(f"  Mean PSNR          : {report.mean_psnr or 'N/A (no reference)'} dB")
        print(f"  Mean SSIM          : {report.mean_ssim or 'N/A'}")
        print(f"  Mean LPIPS         : {report.mean_lpips or 'N/A'}")
        print(f"  Temporal Consist.  : {report.temporal_consistency:.5f} (lower=better)")
        print(f"  Mean Hallu. Frac.  : {report.mean_hallucinated_fraction:.2%}")
        print(f"  No-hallu. indicator: {report.no_hallucination_indicator:.2%} frames clean")
        print("=" * 45)
