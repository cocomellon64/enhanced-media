# Forensics & Audit Mode

## Purpose

Forensics mode is designed for use cases where any hallucination is unacceptable:
- Legal proceedings
- Medical imaging review
- Identity verification (with appropriate caveats)
- Evidentiary media analysis

**Important:** Enhanced media is not equivalent to the original. Even in forensics mode, the model applies processing. All outputs must be reviewed by a qualified human expert before use in legal or medical contexts.

---

## What Forensics Mode Does

| Operation            | Normal Mode          | Forensics Mode              |
|----------------------|----------------------|-----------------------------|
| Sharpening           | Edge-aware, moderate | Disabled (0 strength)       |
| Deblurring           | Adaptive strength    | Max 0.4 strength, heavy reg |
| Denoising            | Adaptive             | Conservative (spatial only) |
| Temporal smoothing   | Enabled              | Disabled                    |
| SR hallucination     | Allowed with flagging| Penalized; flagged always   |
| Color correction     | Conservative         | None (passthrough)          |
| Confidence threshold | 0.40                 | 0.60 (stricter flagging)    |

---

## Processing Manifest

Every output in forensics mode includes a `_manifest.json` file containing:

```json
{
  "manifest_version": "1.0",
  "model_version": "<semver>",
  "model_checksum": "<sha256 fingerprint>",
  "mode": "forensics",
  "config": { ... all parameters ... },
  "metrics": {
    "temporal_consistency_score": 0.0,
    "hallucinated_frames_count": 0
  },
  "warnings": [],
  "hallucinated_frame_indices": []
}
```

---

## Confidence Map Interpretation

- **Green (high confidence, > 0.8):** Model is confident the output matches the original content.
- **Yellow (medium confidence, 0.4–0.8):** Model applied learned enhancement; region may differ slightly from original.
- **Red (low confidence, < 0.4):** Model output in this region is uncertain; may not reflect original content. Treat with caution.

In forensics mode, regions marked red must be annotated in any report using the output.

---

## CLI Example

```bash
python scripts/enhance_cli.py image evidence.jpg evidence_enhanced.png \
  --mode forensics \
  --save-confidence \
  --save-manifest \
  --save-diff
```

Outputs:
- `evidence_enhanced.png` — enhanced image
- `evidence_enhanced_confidence.npy` — raw float32 confidence map
- `evidence_enhanced_confidence.png` — color-coded confidence map
- `evidence_enhanced_manifest.json` — full audit manifest
- `evidence_enhanced_diff.png` — pixel difference heatmap

---

## What the Model Cannot Do

- It cannot reconstruct content that was not captured by the camera sensor.
- It cannot remove compression artifacts without risk of altering fine details.
- It cannot determine the original scene from a heavily degraded image with certainty.
- It should never be used as the sole basis for identification in legal proceedings.
