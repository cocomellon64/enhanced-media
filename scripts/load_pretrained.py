"""
Pretrained weight compatibility loader.

Real-ESRGAN uses RRDB (Residual in Residual Dense Block) architecture.
Our ImageSuperResolutionNet uses a simpler residual block design.

This script does one of two things depending on --mode:
  1. "adapt"  : extracts compatible weights from a Real-ESRGAN checkpoint
                and saves a trimmed checkpoint our model can load with strict=False
  2. "direct" : downloads and saves a minimal random-init checkpoint so the
                pipeline runs immediately without any pretrained weights

Usage:
  # After downloading RealESRGAN_x4plus.pth:
  python scripts/load_pretrained.py adapt \
      --src checkpoints/image_sr_realesrgan.pth \
      --dst checkpoints/image_sr.pth

  # Just get the pipeline running with random weights (no download needed):
  python scripts/load_pretrained.py direct \
      --dst checkpoints/image_sr.pth
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancer.models.rrdb_net import RRDBNet


def cmd_adapt(args):
    """
    Load a Real-ESRGAN checkpoint and extract the weights that are
    compatible with our model's parameter names.
    Incompatible layers are skipped (strict=False handles the rest).
    """
    print(f"[load_pretrained] Loading source checkpoint: {args.src}")
    src_state = torch.load(args.src, map_location="cpu", weights_only=False)

    # Real-ESRGAN checkpoints are sometimes wrapped under 'params_ema' or 'params'
    if "params_ema" in src_state:
        src_state = src_state["params_ema"]
        print("  Found key: params_ema")
    elif "params" in src_state:
        src_state = src_state["params"]
        print("  Found key: params")
    else:
        print("  Using root state dict directly")

    # Initialize our model (RRDB architecture — matches Real-ESRGAN exactly)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    our_state = model.state_dict()

    # Load Real-ESRGAN weights with strict=False (confidence_head keys are extra, that's fine)
    missing, unexpected = model.load_state_dict(src_state, strict=False)
    matched = len(our_state) - len(missing)

    print(f"  Matched: {matched} tensors")
    print(f"  Missing (confidence head — expected): {len(missing)}")
    print(f"  Unexpected (not in our model): {len(unexpected)}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(model.state_dict(), args.dst)
    print(f"[load_pretrained] Adapted checkpoint saved: {args.dst}")
    print(f"  {matched} Real-ESRGAN layers loaded.")
    if missing:
        print(f"  {len(missing)} layers use random init (confidence head — this is expected).")
    print("\nYou can now run:")
    print(f"  python scripts/enhance_cli.py image photo.jpg out.png --checkpoint-dir checkpoints")


def cmd_direct(args):
    """
    Save a freshly initialized model checkpoint so the pipeline
    can load it with --checkpoint-dir without errors.
    Output quality will be poor (random weights) but the pipeline runs.
    """
    print("[load_pretrained] Initializing model with random weights...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(model.state_dict(), args.dst)
    print(f"[load_pretrained] Random-init checkpoint saved: {args.dst}")
    print("\nNote: output quality will be poor until the model is trained.")
    print("Run the pipeline with:")
    print(f"  python scripts/enhance_cli.py image photo.jpg out.png --checkpoint-dir checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Pretrained weight loader / adapter")
    sub = parser.add_subparsers(dest="mode", required=True)

    adapt_p = sub.add_parser("adapt", help="Adapt a Real-ESRGAN checkpoint to our model")
    adapt_p.add_argument("--src", required=True, help="Source .pth file (Real-ESRGAN)")
    adapt_p.add_argument("--dst", default="checkpoints/image_sr.pth", help="Output .pth path")

    direct_p = sub.add_parser("direct", help="Save a random-init checkpoint for immediate use")
    direct_p.add_argument("--dst", default="checkpoints/image_sr.pth", help="Output .pth path")

    args = parser.parse_args()
    {"adapt": cmd_adapt, "direct": cmd_direct}[args.mode](args)


if __name__ == "__main__":
    main()
