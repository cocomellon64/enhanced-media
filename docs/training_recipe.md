# Training Recipe

## Overview

Training proceeds in four stages. Each stage builds on the previous.
All runs emit an audit log in `training_logs/` and versioned checkpoints in `checkpoints/`.

---

## Stage 1: Image Super-Resolution (ImageSuperResolutionNet)

**Goal:** Learn to upscale 4× from diverse degraded inputs with minimal hallucination.

**Dataset:** DIV2K (800 HR training images) + Flickr2K (2650 images).  
Download: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Degradation augmentation (applied on-the-fly):**
- Random Gaussian blur (σ 0.5–3.0)
- Random motion blur (kernel 5–15px)
- JPEG compression (quality 30–95)
- Gaussian noise (σ 0–25)
- 4× bicubic downscale

**Loss:**
```
L = 1.0 × Charbonnier(pred, GT)
  + 0.1 × Perceptual_VGG(pred, GT)   [relu3_4]
  + 0.5 × SSIM_loss(pred, GT)
```

Perceptual weight is intentionally low (0.1) to prioritize fidelity over texture richness.

**Hyperparameters:**
| Param           | Value   |
|-----------------|---------|
| LR              | 2e-4    |
| Batch size      | 8       |
| Patch size (HR) | 128×128 |
| Epochs          | 100     |
| Scheduler       | CosineAnnealing |
| Optimizer       | Adam (β₁=0.9, β₂=0.999) |
| Grad clip       | norm 1.0 |

**Validation:** Hold out 100 DIV2K images. Monitor PSNR and hallucinated fraction.

---

## Stage 2: Video Super-Resolution (VideoSuperResolutionNet)

**Dataset:** REDS (Sharp+Blur split) + Vimeo-90K.  
- REDS: https://seungjunnah.github.io/Datasets/reds.html  
- Vimeo-90K: http://toflow.csie.ntu.edu.tw/

**Loss:** Stage 1 losses + temporal consistency term:
```
L += 0.2 × TemporalConsistencyLoss(t, t-1, flow)
```

**Additional training notes:**
- Pre-train optical flow estimator on Flying Chairs / FlyingThings3D before joint training.
- Freeze optical flow during first 10 epochs.
- Enable joint fine-tuning with LR 5e-5 after epoch 10.

---

## Stage 3: Deblurring (DeblurNet)

**Dataset:** GoPro Sharp/Blur dataset.  
Download: https://github.com/SeungjunNah/DeepDeblur-PyTorch

**Loss:**
```
L = L1(deblurred, sharp) + 0.05 × Perceptual(deblurred, sharp)
```

**Key constraint:** PSF confidence gates strength. If confidence < 0.5, the model cannot apply more than 50% strength regardless of the network output. This prevents aggressive deconvolution on uncertain estimates.

---

## Stage 4: Joint Fine-tuning

Fine-tune the full pipeline end-to-end on real-world test videos (with manual quality labels).
Use a small LR (5e-6) and early stopping.

---

## Model Update Policy (Learning from User Feedback)

1. Collect explicit labeled feedback only (accept / reject / region correction).
2. Apply conservative fine-tuning: LR ≤ 5e-6, ≤ 5 epochs per update batch.
3. Validate against a held-out fidelity test set before deploying updates.
4. Each update increments model version and records: update timestamp, feedback count, training data checksum.
5. Users can opt out of data collection entirely — their feedback is discarded.

---

## Audit Trail

Every training run produces:
- `training_logs/<model>_training_log.json` — per-step metrics
- `checkpoints/<model>_epoch<N>_meta.json` — checkpoint metadata with dataset hash
- `checkpoints/<model>.pth` — best checkpoint (by validation loss)
