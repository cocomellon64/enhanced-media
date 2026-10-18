# Video & Image Enhancer AI

Conservative, auditable media enhancement to 4K. Accepts images and videos, outputs enhanced media with per-frame confidence maps, hallucination detection, and full processing logs.

---

## Features

- **4K upscaling** via learned super-resolution (4× scale)
- **Edge-aware sharpening** — no halos, no ringing
- **Temporal denoising** — flicker-free video enhancement
- **Motion-compensated multi-frame SR** — recovers sub-pixel detail from video
- **Blind deblurring** with PSF confidence gating
- **Confidence maps** per frame — flags uncertain/hallucinated regions
- **Forensics mode** — zero hallucination, full audit trail
- **REST API** (FastAPI) + **Web UI** (Gradio) + **CLI**

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
# or
pip install -e ".[all]"
```

### Enhance an Image (CLI)

```bash
python scripts/enhance_cli.py image photo.jpg enhanced.png --mode balanced
```

### Enhance a Video

```bash
python scripts/enhance_cli.py video clip.mp4 clip_4k.mp4 --mode quality --save-manifest
```

### Forensics Mode (Legal / Evidentiary Use)

```bash
python scripts/enhance_cli.py image evidence.jpg evidence_enhanced.png \
  --mode forensics \
  --save-confidence \
  --save-manifest \
  --save-diff
```

### Launch the Web UI

```bash
python -m enhancer.ui.app
# Open: http://localhost:7860
```

### Launch the REST API

```bash
python -m enhancer.api.server
# POST to: http://localhost:8000/enhance
```

---

## Enhancement Modes

| Mode          | Sharpen | Denoise | Hallucination | Speed  |
|---------------|---------|---------|---------------|--------|
| conservative  | Low     | Moderate| Strictly flagged | Fast |
| balanced      | Medium  | Medium  | Flagged        | Medium |
| quality       | High    | Strong  | Flagged        | Slow   |
| fast          | Low     | Light   | Flagged        | Fast   |
| forensics     | None    | Light   | **Zero tolerance** | Medium |

---

## Pipeline Stages

1. **Preprocess** — blur type, noise level, shake detection
2. **Stabilize** — optional, if shake detected
3. **Denoise** — temporal-aware, edge-preserving
4. **Optical flow** — motion estimation for multi-frame fusion
5. **Super-resolution** — multi-frame (video) or single-image SR
6. **Deblur** — PSF-confidence-gated blind deblurring
7. **Per-frame refinement** — edge-aware unsharp mask
8. **Temporal smoothing** — EMA-based flicker suppression
9. **Postprocess** — conservative color/exposure correction + artifact suppression
10. **Output** — enhanced media + confidence maps + processing log + manifest

---

## Output Artifacts

For every processed file:
- Enhanced image/video
- `_confidence.png` — color-coded confidence map (green=high, red=uncertain)
- `_confidence.npy` — raw float32 confidence data
- `_manifest.json` — model version, parameters, checksum, hallucination summary
- `_diff.png` — pixel difference heatmap (optional)

---

## API Contract

See [docs/api_spec.md](docs/api_spec.md) for the full REST API specification.

---

## Training

See [docs/training_recipe.md](docs/training_recipe.md) for the full training recipe.

```bash
python scripts/train.py image_sr --data-dir ./data/DIV2K_HR --epochs 100
```

Recommended datasets: [docs/training_datasets.md](docs/training_datasets.md)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Forensics & Legal Use

See [docs/forensics_audit_mode.md](docs/forensics_audit_mode.md) for forensics mode documentation and important limitations.

**Important:** Enhanced media is a processed artifact. Forensics mode minimizes hallucination but does not guarantee perfect fidelity. Always have qualified experts review outputs used in legal, medical, or evidentiary contexts.

---

## Project Structure

```
video_image_enhancer/
├── enhancer/
│   ├── models/           # SR, deblur, denoiser, optical flow networks
│   ├── pipeline/         # main pipeline, pre/post-processing, temporal smoother
│   ├── api/              # FastAPI REST server
│   ├── ui/               # Gradio web interface
│   ├── evaluation/       # PSNR, SSIM, LPIPS, temporal consistency
│   └── utils/            # confidence maps, visual diff, media I/O
├── scripts/
│   ├── enhance_cli.py    # CLI
│   └── train.py          # training recipe
├── tests/                # pytest test suite
├── docs/                 # training recipe, API spec, forensics guide
├── configs/              # default hyperparameters per mode
└── requirements.txt
```
