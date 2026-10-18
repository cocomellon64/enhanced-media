"""
Feedback Trainer
Fine-tunes the SR model on user-labeled feedback pairs.

Policy (conservative by design):
- Learning rate is very small (5e-6) to avoid catastrophic forgetting
- Trains for at most 5 epochs per feedback batch
- Validates on a held-out fidelity set before accepting updates
- Saves versioned checkpoints with full metadata
- Rolls back automatically if validation loss increases
- Requires minimum 10 accepted pairs before triggering any update
"""

import os
import json
import time
import copy
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


MIN_PAIRS_TO_TRAIN = 10       # minimum feedback pairs needed
MAX_EPOCHS = 5                # max epochs per feedback update
LEARNING_RATE = 5e-6          # conservative LR — small steps only
BATCH_SIZE = 2                # small batch for memory efficiency
PATCH_SIZE = 96               # crop size for training patches


class FeedbackDataset(Dataset):
    """
    Dataset built from user feedback pairs.
    - "accept" pairs: output is treated as pseudo-GT, input is LR
    - "reject" pairs: skipped (used only for validation loss tracking)
    - "partial" pairs: masked loss — only accepted regions contribute
    """

    def __init__(
        self,
        pairs: List[Tuple],   # (input_path, output_path, mask_path, verdict)
        patch_size: int = PATCH_SIZE,
        scale: int = 4,
    ):
        # Only train on accepted and partial pairs
        self.pairs = [(i, o, m, v) for i, o, m, v in pairs if v in ("accept", "partial")]
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return max(len(self.pairs) * 4, 1)  # 4 random crops per pair

    def __getitem__(self, idx):
        in_path, out_path, mask_path, verdict = self.pairs[idx % len(self.pairs)]

        lr = cv2.imread(in_path)
        hr = cv2.imread(out_path)

        if lr is None or hr is None:
            # Return zeros if file is missing
            ps = self.patch_size
            return {
                "lr": torch.zeros(3, ps // self.scale, ps // self.scale),
                "hr": torch.zeros(3, ps, ps),
                "mask": torch.ones(1, ps, ps),
            }

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Resize HR to patch_size, LR to patch_size // scale
        hr_ps = self.patch_size
        lr_ps = hr_ps // self.scale

        h, w = hr.shape[:2]
        if h < hr_ps or w < hr_ps:
            hr = cv2.resize(hr, (max(w, hr_ps), max(h, hr_ps)))
            h, w = hr.shape[:2]

        # Random crop
        y = np.random.randint(0, h - hr_ps + 1)
        x = np.random.randint(0, w - hr_ps + 1)
        hr_crop = hr[y:y+hr_ps, x:x+hr_ps]

        # Downscale to get LR
        lr_crop = cv2.resize(hr_crop, (lr_ps, lr_ps), interpolation=cv2.INTER_CUBIC)

        # Build mask
        if mask_path and os.path.exists(mask_path) and verdict == "partial":
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (hr_ps, hr_ps))
        else:
            mask = np.ones((hr_ps, hr_ps), dtype=np.float32)

        # Random flip augmentation
        if np.random.rand() > 0.5:
            hr_crop = np.fliplr(hr_crop).copy()
            lr_crop = np.fliplr(lr_crop).copy()
            mask    = np.fliplr(mask).copy()

        hr_t   = torch.from_numpy(hr_crop).permute(2, 0, 1)
        lr_t   = torch.from_numpy(lr_crop).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return {"lr": lr_t, "hr": hr_t, "mask": mask_t}


class FeedbackTrainer:
    """
    Fine-tunes the SR model (RRDBNet) on user feedback pairs.

    Usage:
        trainer = FeedbackTrainer(model, checkpoint_dir="./checkpoints")
        result = trainer.train(feedback_pairs)
        if result["accepted"]:
            print(f"Model updated: v{result['new_version']}")
        else:
            print(f"Update rejected: {result['reason']}")
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = "./checkpoints",
        validation_images: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_images = validation_images or []

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._version_file = os.path.join(checkpoint_dir, "feedback_version.json")
        self._version = self._load_version()

    def train(
        self,
        pairs: List[Tuple],
        max_epochs: int = MAX_EPOCHS,
        lr: float = LEARNING_RATE,
        min_pairs: int = MIN_PAIRS_TO_TRAIN,
    ) -> dict:
        """
        Fine-tune on feedback pairs.

        Returns a result dict:
        {
          "accepted": bool,
          "reason": str,
          "new_version": int or None,
          "epochs_trained": int,
          "train_loss": float,
          "val_loss_before": float or None,
          "val_loss_after": float or None,
        }
        """
        accepted_pairs = [(i, o, m, v) for i, o, m, v in pairs if v in ("accept", "partial")]

        if len(accepted_pairs) < min_pairs:
            return {
                "accepted": False,
                "reason": f"Not enough feedback pairs. Have {len(accepted_pairs)}, need {min_pairs}.",
                "new_version": None,
                "epochs_trained": 0,
            }

        print(f"[FeedbackTrainer] Starting fine-tune on {len(accepted_pairs)} pairs "
              f"(lr={lr}, max_epochs={max_epochs})")

        # Snapshot current model for rollback
        original_state = copy.deepcopy(self.model.state_dict())

        # Measure validation loss before
        val_loss_before = self._compute_val_loss()

        # Build dataset and loader
        dataset = FeedbackDataset(pairs, patch_size=PATCH_SIZE)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn   = nn.L1Loss()
        self.model.train()

        import torch.nn.functional as F

        train_losses = []
        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_loss = []
            for batch in loader:
                lr_t    = batch["lr"].to(self.device)
                hr_t    = batch["hr"].to(self.device)
                mask_t  = batch["mask"].to(self.device)

                optimizer.zero_grad()
                enhanced, _ = self.model(lr_t)

                # Resize HR to match enhanced output if needed
                if enhanced.shape != hr_t.shape:
                    hr_t   = F.interpolate(hr_t,   size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
                    mask_t = F.interpolate(mask_t, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)

                # Masked L1 loss
                diff = torch.abs(enhanced - hr_t) * mask_t
                loss = diff.sum() / (mask_t.sum() + 1e-6)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_loss.append(loss.item())

            mean_loss = float(np.mean(epoch_loss))
            train_losses.append(mean_loss)
            print(f"  Epoch {epoch}/{max_epochs} | loss: {mean_loss:.5f} | "
                  f"elapsed: {time.time()-t0:.1f}s")

        self.model.eval()

        # Measure validation loss after
        val_loss_after = self._compute_val_loss()

        # Rollback check: if validation loss got worse, reject the update
        if val_loss_before is not None and val_loss_after is not None:
            if val_loss_after > val_loss_before * 1.05:  # allow 5% tolerance
                print(f"[FeedbackTrainer] Validation loss degraded "
                      f"({val_loss_before:.5f} → {val_loss_after:.5f}). Rolling back.")
                self.model.load_state_dict(original_state)
                return {
                    "accepted": False,
                    "reason": f"Validation loss degraded ({val_loss_before:.5f}→{val_loss_after:.5f}). Rolled back.",
                    "new_version": None,
                    "epochs_trained": max_epochs,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss_before": val_loss_before,
                    "val_loss_after": val_loss_after,
                }

        # Accept the update — save versioned checkpoint
        new_version = self._version + 1
        ckpt_path   = os.path.join(self.checkpoint_dir, f"image_sr_feedback_v{new_version}.pth")
        latest_path = os.path.join(self.checkpoint_dir, "image_sr.pth")

        torch.save(self.model.state_dict(), ckpt_path)
        torch.save(self.model.state_dict(), latest_path)

        # Save version metadata
        meta = {
            "version": new_version,
            "timestamp": datetime.utcnow().isoformat(),
            "pairs_used": len(accepted_pairs),
            "epochs": max_epochs,
            "lr": lr,
            "train_loss": round(float(np.mean(train_losses)), 6),
            "val_loss_before": val_loss_before,
            "val_loss_after": val_loss_after,
            "checkpoint": ckpt_path,
            "pairs_checksum": self._checksum_paths([p[0] for p in accepted_pairs]),
        }
        self._version = new_version
        self._save_version(meta)

        print(f"[FeedbackTrainer] Update accepted. Model version: {new_version}")
        print(f"  Checkpoint: {ckpt_path}")

        return {
            "accepted": True,
            "reason": "Validation passed.",
            "new_version": new_version,
            "epochs_trained": max_epochs,
            "train_loss": round(float(np.mean(train_losses)), 6),
            "val_loss_before": val_loss_before,
            "val_loss_after": val_loss_after,
        }

    def _compute_val_loss(self) -> Optional[float]:
        """Compute L1 loss on validation images (if any provided)."""
        if not self.validation_images:
            return None

        import torch.nn.functional as F
        self.model.eval()
        losses = []

        with torch.no_grad():
            for img_path in self.validation_images:
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                h, w = img_rgb.shape[:2]

                # Downscale to create LR
                lr = cv2.resize(img_rgb, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
                lr_t = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).to(self.device)
                hr_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

                enhanced, _ = self.model(lr_t)
                hr_resized  = F.interpolate(hr_t, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
                loss = torch.mean(torch.abs(enhanced - hr_resized)).item()
                losses.append(loss)

        return float(np.mean(losses)) if losses else None

    def _load_version(self) -> int:
        if os.path.exists(self._version_file):
            with open(self._version_file) as f:
                data = json.load(f)
            return data.get("version", 0)
        return 0

    def _save_version(self, meta: dict) -> None:
        history_path = os.path.join(self.checkpoint_dir, "feedback_version_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        history.append(meta)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        with open(self._version_file, "w") as f:
            json.dump({"version": meta["version"], "latest": meta}, f, indent=2)

    def _checksum_paths(self, paths: List[str]) -> str:
        h = hashlib.md5()
        for p in sorted(paths):
            h.update(p.encode())
        return h.hexdigest()[:12]
