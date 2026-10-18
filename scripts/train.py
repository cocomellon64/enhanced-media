"""
Training recipe for the super-resolution, deblur, and denoiser models.

Training strategy:
- Stage 1: Train ImageSR with L1 + perceptual loss (no adversarial)
- Stage 2: Fine-tune VideoSR with temporal consistency loss
- Stage 3: Train DeblurNet with synthetic blur augmentation
- Stage 4: Fine-tune jointly on real-world degradation dataset

Loss functions:
- L1 / Charbonnier: pixel fidelity (suppress hallucination)
- Perceptual (VGG): recover texture detail without ringing
- SSIM: structural consistency
- Temporal: flow-warp consistency between consecutive frames
- Fidelity regularizer: penalizes deviation from bicubic baseline

All training runs log metrics and checkpoint versions for audit trail.
"""

import os
import sys
import json
import time
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Smooth L1 variant that's less sensitive to outliers."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for texture fidelity.
    Uses relu3_4 features from VGG19.
    Falls back to L1 if torchvision not available.
    """
    def __init__(self, layer: str = "relu3_4"):
        super().__init__()
        self.layer = layer
        self._vgg = None
        self._available = False
        try:
            import torchvision.models as models
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            # relu3_4 is after layer index 18 in features
            self._vgg = nn.Sequential(*list(vgg.features)[:19]).eval()
            for p in self._vgg.parameters():
                p.requires_grad = False
            self._available = True
        except (ImportError, Exception) as e:
            print(f"[PerceptualLoss] torchvision not available, falling back to L1: {e}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return torch.mean(torch.abs(pred - target))
        device = pred.device
        self._vgg = self._vgg.to(device)
        feat_pred = self._vgg(pred.clamp(0, 1))
        feat_tgt = self._vgg(target.clamp(0, 1))
        return torch.mean((feat_pred - feat_tgt) ** 2)


class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes temporal flicker: warped previous frame should match current frame
    in non-occluded regions.
    """
    def forward(
        self,
        current: torch.Tensor,
        prev_enhanced: torch.Tensor,
        flow: torch.Tensor,
        occlusion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        B, C, H, W = current.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=current.device),
            torch.arange(W, dtype=torch.float32, device=current.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        warped_grid = grid + flow
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / max(W - 1, 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / max(H - 1, 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)

        warped_prev = F.grid_sample(prev_enhanced, warped_grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
        diff = torch.abs(current - warped_prev)
        if occlusion_mask is not None:
            diff = diff * (1 - occlusion_mask)
        return diff.mean()


class CombinedSRLoss(nn.Module):
    """
    Combined loss for super-resolution training.
    Weights are configurable; defaults favor fidelity (conservative).
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.5,
    ):
        super().__init__()
        self.l1 = CharbonnierLoss()
        self.perceptual = PerceptualLoss()
        self.weights = {
            "l1": l1_weight,
            "perceptual": perceptual_weight,
            "ssim": ssim_weight,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l1 = self.l1(pred, target)
        perc = self.perceptual(pred, target)
        total = self.weights["l1"] * l1 + self.weights["perceptual"] * perc
        return {"total": total, "l1": l1.item(), "perceptual": perc.item()}


# ---------------------------------------------------------------------------
# Synthetic degradation dataset
# ---------------------------------------------------------------------------

class SyntheticDegradationDataset(Dataset):
    """
    Generates synthetic LR-HR pairs from HR images by applying:
    - Gaussian / motion blur
    - JPEG compression artifacts
    - Gaussian noise
    - Bicubic downscaling

    Use real-world datasets (DIV2K, REDS, Vimeo-90K) for actual training.
    See docs/training_datasets.md for download instructions.
    """

    def __init__(
        self,
        hr_image_dir: str,
        scale: int = 4,
        patch_size: int = 128,
        augment: bool = True,
    ):
        import glob
        self.hr_paths = sorted(glob.glob(os.path.join(hr_image_dir, "*.png")) +
                               glob.glob(os.path.join(hr_image_dir, "*.jpg")))
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        if not self.hr_paths:
            raise ValueError(f"No images found in {hr_image_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths) * 4  # 4 random crops per image

    def __getitem__(self, idx: int) -> dict:
        import cv2
        hr_path = self.hr_paths[idx % len(self.hr_paths)]
        hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if hr is None:
            raise IOError(f"Cannot read {hr_path}")

        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random crop
        h, w = hr.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            hr = cv2.resize(hr, (max(w, ps), max(h, ps)))
            h, w = hr.shape[:2]
        y = np.random.randint(0, h - ps + 1)
        x = np.random.randint(0, w - ps + 1)
        hr_patch = hr[y:y+ps, x:x+ps].copy()

        # Augmentation: flip/rotate
        if self.augment:
            if np.random.rand() > 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            if np.random.rand() > 0.5:
                hr_patch = np.flipud(hr_patch).copy()

        # Synthetic degradation
        lr_patch = self._degrade(hr_patch)

        hr_t = torch.from_numpy(hr_patch).permute(2, 0, 1)
        lr_t = torch.from_numpy(lr_patch).permute(2, 0, 1)
        return {"hr": hr_t, "lr": lr_t}

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        """Apply random degradation pipeline."""
        import cv2
        deg = img.copy()

        # Random blur
        blur_type = np.random.choice(["none", "gaussian", "motion"], p=[0.3, 0.4, 0.3])
        if blur_type == "gaussian":
            k = np.random.choice([3, 5, 7])
            deg = cv2.GaussianBlur(deg, (k, k), 0)
        elif blur_type == "motion":
            # Horizontal motion blur
            k = np.random.randint(5, 15)
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            deg = cv2.filter2D(deg, -1, kernel)

        # Downscale then upscale
        h, w = deg.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        deg = cv2.resize(deg, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        # Keep at LR size (SR model will upsample)

        # Noise
        if np.random.rand() > 0.3:
            sigma = np.random.uniform(0, 25) / 255.0
            deg = deg + np.random.randn(*deg.shape).astype(np.float32) * sigma

        return np.clip(deg, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class AuditLogger:
    """Records training metrics and model updates with versioned history."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []

    def log(self, epoch: int, step: int, metrics: dict, extra: dict = None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)

    def save(self, filename: str = "training_log.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def checkpoint_metadata(self, model_name: str, epoch: int, params: dict) -> str:
        """Create versioned checkpoint metadata."""
        meta = {
            "model": model_name,
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": params,
        }
        path = os.path.join(self.log_dir, f"{model_name}_epoch{epoch:04d}_meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        return path


def train_image_sr(args):
    """Stage 1: Train ImageSuperResolutionNet."""
    from enhancer.models.super_resolution import ImageSuperResolutionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] ImageSR on {device}")

    model = ImageSuperResolutionNet(scale=args.scale, num_blocks=args.num_blocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedSRLoss(l1_weight=1.0, perceptual_weight=0.1)

    dataset = SyntheticDegradationDataset(args.data_dir, scale=args.scale, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    audit = AuditLogger(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()
            enhanced, confidence = model(lr)

            # Upsample HR to match enhanced shape for loss
            if enhanced.shape != hr.shape:
                import torch.nn.functional as F
                hr_up = F.interpolate(hr, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
            else:
                hr_up = hr

            losses = loss_fn(enhanced, hr_up)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(losses["total"].item())
            step += 1

            if step % 100 == 0:
                mean_loss = np.mean(epoch_losses[-100:])
                print(f"  Epoch {epoch}/{args.epochs} Step {step} | Loss: {mean_loss:.4f}")
                audit.log(epoch, step, {
                    "loss": round(mean_loss, 5),
                    "l1": round(losses["l1"], 5),
                    "perceptual": round(losses["perceptual"], 5),
                })

        scheduler.step()
        mean_epoch_loss = np.mean(epoch_losses)

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"image_sr_epoch{epoch:04d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        audit.checkpoint_metadata("image_sr", epoch, {"loss": mean_epoch_loss, "lr": optimizer.param_groups[0]["lr"]})

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "image_sr.pth"))
            print(f"  [✓] New best checkpoint saved (loss: {best_loss:.4f})")

    audit.save("image_sr_training_log.json")
    print(f"[Train] ImageSR training complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train enhancement models")
    parser.add_argument("stage", choices=["image_sr", "video_sr", "deblur", "denoiser"],
                        help="Which model stage to train")
    parser.add_argument("--data-dir", required=True, help="Path to HR training images")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--log-dir", default="./training_logs", help="Where to save audit logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=4, help="SR upscale factor")
    parser.add_argument("--patch-size", type=int, default=128, help="Training patch size (HR)")
    parser.add_argument("--num-blocks", type=int, default=16, help="Residual blocks in SR model")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    dispatch = {
        "image_sr": train_image_sr,
    }
    fn = dispatch.get(args.stage)
    if fn is None:
        print(f"Stage '{args.stage}' training not yet implemented in this scaffold.")
        print("See docs/training_recipe.md for the full training recipe.")
    else:
        fn(args)


if __name__ == "__main__":
    main()
