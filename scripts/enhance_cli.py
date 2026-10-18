#!/usr/bin/env python3
"""
Command-line interface for the Video & Image Enhancer AI

Usage examples:
  # Enhance an image (balanced mode, 4K output)
  python scripts/enhance_cli.py image input.jpg output.png

  # Enhance with forensics mode (no hallucination)
  python scripts/enhance_cli.py image input.jpg output.png --mode forensics

  # Enhance a video
  python scripts/enhance_cli.py video input.mp4 output.mp4 --mode quality

  # Preview first frame of a video
  python scripts/enhance_cli.py preview input.mp4 preview.png

  # Batch enhance all images in a folder
  python scripts/enhance_cli.py batch ./input_dir ./output_dir --mode balanced

  # Evaluate against reference dataset
  python scripts/enhance_cli.py evaluate ./originals ./enhanced ./references report.json
"""

import argparse
import os
import sys
import json
import time
import glob
import cv2
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.pipeline.enhancer_pipeline import EnhancerPipeline, EnhancementConfig
from enhancer.utils.confidence_map import save_confidence_map, overlay_hallucination_mask
from enhancer.utils.visual_diff import generate_visual_diff
from enhancer.utils.media_io import SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS
from enhancer.evaluation.evaluator import Evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video & Image Enhancer AI - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args shared across commands
    def add_common(p):
        p.add_argument("--mode", choices=["fast", "balanced", "quality", "conservative", "forensics"],
                       default="balanced", help="Enhancement mode")
        p.add_argument("--aggressive", action="store_true", help="Enable aggressive mode (explicit opt-in)")
        p.add_argument("--sharpen", type=float, default=0.5, metavar="[0-1]", help="Sharpen amount")
        p.add_argument("--denoise", type=float, default=0.5, metavar="[0-1]", help="Denoise strength")
        p.add_argument("--temporal", type=float, default=0.4, metavar="[0-1]", help="Temporal smoothing")
        p.add_argument("--width", type=int, default=3840, help="Target output width (default: 3840)")
        p.add_argument("--height", type=int, default=2160, help="Target output height (default: 2160)")
        p.add_argument("--no-aspect", action="store_true", help="Stretch to exact target resolution")
        p.add_argument("--no-stabilize", action="store_true", help="Disable stabilization")
        p.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or None for auto")
        p.add_argument("--checkpoint-dir", default=None, help="Path to pretrained model checkpoints")
        p.add_argument("--save-confidence", action="store_true", help="Save confidence maps alongside output")
        p.add_argument("--save-diff", action="store_true", help="Save visual diff (original vs enhanced)")
        p.add_argument("--save-manifest", action="store_true", help="Save processing manifest JSON")

    # image subcommand
    img_p = sub.add_parser("image", help="Enhance a single image")
    img_p.add_argument("input", help="Input image path")
    img_p.add_argument("output", help="Output image path")
    add_common(img_p)

    # video subcommand
    vid_p = sub.add_parser("video", help="Enhance a video file")
    vid_p.add_argument("input", help="Input video path")
    vid_p.add_argument("output", help="Output video path")
    add_common(vid_p)

    # preview subcommand
    prev_p = sub.add_parser("preview", help="Quick preview (first frame or image thumbnail)")
    prev_p.add_argument("input", help="Input image or video path")
    prev_p.add_argument("output", help="Output preview PNG path")
    add_common(prev_p)

    # batch subcommand
    batch_p = sub.add_parser("batch", help="Batch enhance all images/videos in a directory")
    batch_p.add_argument("input_dir", help="Input directory")
    batch_p.add_argument("output_dir", help="Output directory")
    add_common(batch_p)

    # evaluate subcommand
    eval_p = sub.add_parser("evaluate", help="Compute quality metrics")
    eval_p.add_argument("originals_dir", help="Directory of original LR images")
    eval_p.add_argument("enhanced_dir", help="Directory of enhanced output images")
    eval_p.add_argument("--references-dir", default=None, help="Directory of HR ground truth images")
    eval_p.add_argument("output_report", help="Output JSON report path")

    return parser


def build_config(args) -> EnhancementConfig:
    return EnhancementConfig(
        target_width=args.width,
        target_height=args.height,
        mode=args.mode,
        aggressive=args.aggressive,
        stabilize=not args.no_stabilize,
        preserve_aspect=not args.no_aspect,
        sharpen_amount=args.sharpen,
        denoise_strength=args.denoise,
        temporal_smoothing=args.temporal,
    )


def build_pipeline(args) -> EnhancerPipeline:
    config = build_config(args)
    pipeline = EnhancerPipeline(config=config, device=getattr(args, "device", None))
    pipeline.load_models(checkpoint_dir=getattr(args, "checkpoint_dir", None))
    return pipeline


def cmd_image(args):
    print(f"[CLI] Enhancing image: {args.input}")
    pipeline = build_pipeline(args)
    img = cv2.imread(args.input)
    if img is None:
        print(f"ERROR: Cannot read {args.input}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    enhanced, conf_map, log = pipeline.enhance_image(img)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, enhanced)

    print(f"[CLI] Enhanced saved: {args.output} ({elapsed:.2f}s)")
    print(f"      Output resolution: {enhanced.shape[1]}x{enhanced.shape[0]}")
    print(f"      Hallucinated fraction: {log.get('hallucinated_fraction', 0):.2%}")

    _save_extras(args, img, enhanced, conf_map, log, args.output)


def cmd_video(args):
    print(f"[CLI] Enhancing video: {args.input}")
    pipeline = build_pipeline(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def progress(idx, total, fps_est):
        bar = "#" * int(30 * idx / max(total, 1))
        print(f"\r  [{bar:<30}] {idx}/{total} frames | {fps_est:.1f} fps", end="", flush=True)

    plog = pipeline.enhance_video(args.input, args.output, callback=progress)
    print()  # newline after progress bar

    print(f"[CLI] Enhanced video saved: {args.output}")
    print(f"      Runtime: {plog.total_runtime_s:.1f}s | {plog.metrics.get('effective_fps', 0):.1f} fps")
    print(f"      Temporal consistency: {plog.metrics.get('temporal_consistency_score', 'N/A')}")
    print(f"      Hallucinated frames: {len(plog.hallucinated_frames)}/{plog.total_frames}")

    if args.save_manifest:
        manifest_path = args.output.rsplit(".", 1)[0] + "_manifest.json"
        pipeline.generate_manifest(manifest_path, plog)
        print(f"      Manifest saved: {manifest_path}")


def cmd_preview(args):
    ext = os.path.splitext(args.input)[1].lower()
    pipeline = build_pipeline(args)

    if ext in SUPPORTED_VIDEO_EXTS:
        cap = cv2.VideoCapture(args.input)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("ERROR: Cannot read first frame", file=sys.stderr)
            sys.exit(1)
    else:
        frame = cv2.imread(args.input)

    # Downscale for fast preview
    h, w = frame.shape[:2]
    pw = min(w, 1280)
    ph = int(h * pw / w)
    frame_small = cv2.resize(frame, (pw, ph))

    enhanced, conf_map, log = pipeline.enhance_image(frame_small)
    _, _, composite = generate_visual_diff(frame_small, enhanced, args.output)
    print(f"[CLI] Preview saved: {args.output}")
    print(f"      Hallucinated: {log.get('hallucinated_fraction', 0):.2%}")


def cmd_batch(args):
    exts = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS
    files = [
        f for f in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        print(f"No supported files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = build_pipeline(args)

    print(f"[CLI] Batch processing {len(files)} files...")
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        out_path = os.path.join(args.output_dir, fname)

        print(f"  [{i}/{len(files)}] {fname}", end=" ... ", flush=True)
        t0 = time.time()

        if ext in SUPPORTED_IMAGE_EXTS:
            img = cv2.imread(fpath)
            enhanced, conf_map, log = pipeline.enhance_image(img)
            cv2.imwrite(out_path, enhanced)
            _save_extras(args, img, enhanced, conf_map, log, out_path)
        else:
            plog = pipeline.enhance_video(fpath, out_path)

        print(f"done ({time.time() - t0:.1f}s)")

    print(f"[CLI] Batch complete. Output in: {args.output_dir}")


def cmd_evaluate(args):
    from enhancer.evaluation.evaluator import Evaluator
    exts = SUPPORTED_IMAGE_EXTS

    orig_files = sorted(f for f in glob.glob(os.path.join(args.originals_dir, "*"))
                        if os.path.splitext(f)[1].lower() in exts)
    enh_files = sorted(f for f in glob.glob(os.path.join(args.enhanced_dir, "*"))
                       if os.path.splitext(f)[1].lower() in exts)

    ref_files = None
    if args.references_dir:
        ref_files = sorted(f for f in glob.glob(os.path.join(args.references_dir, "*"))
                           if os.path.splitext(f)[1].lower() in exts)

    originals = [cv2.imread(f) for f in orig_files]
    enhanced = [cv2.imread(f) for f in enh_files]
    references = [cv2.imread(f) for f in ref_files] if ref_files else None

    evaluator = Evaluator()
    report = evaluator.evaluate(originals, enhanced, references=references, dataset_name=args.originals_dir)
    evaluator.save_report(report, args.output_report)


def _save_extras(args, orig, enhanced, conf_map, log, output_path):
    base = output_path.rsplit(".", 1)[0]

    if args.save_confidence:
        save_confidence_map(conf_map, base + "_confidence")

    if args.save_diff:
        _, _, _ = generate_visual_diff(orig, enhanced, base + "_diff.png")
        print(f"      Diff saved: {base}_diff.png")

    if args.save_manifest:
        manifest = {
            "model_version": "1.0.0",
            "mode": args.mode,
            "processing_log": log,
        }
        with open(base + "_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"      Manifest saved: {base}_manifest.json")


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "image": cmd_image,
        "video": cmd_video,
        "preview": cmd_preview,
        "batch": cmd_batch,
        "evaluate": cmd_evaluate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
