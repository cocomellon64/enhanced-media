# API Specification

## Base URL
`http://localhost:8000`

---

## POST /enhance

Enhance an image or video file.

### Request
`multipart/form-data`

| Field               | Type    | Default   | Description                                      |
|---------------------|---------|-----------|--------------------------------------------------|
| `file`              | binary  | required  | Image (JPEG/PNG/TIFF) or video (MP4/MOV/AVI)    |
| `target_width`      | int     | 3840      | Target output width                              |
| `target_height`     | int     | 2160      | Target output height                             |
| `mode`              | string  | balanced  | `fast`, `balanced`, `quality`, `forensics`       |
| `preserve_aspect`   | bool    | true      | Maintain aspect ratio                            |
| `stabilize`         | bool    | true      | Enable camera stabilization                      |
| `aggressive`        | bool    | false     | Enable aggressive enhancement (explicit opt-in)  |
| `sharpen_amount`    | float   | 0.5       | Sharpen strength [0.0–1.0]                       |
| `denoise_strength`  | float   | 0.5       | Denoise strength [0.0–1.0]                       |
| `temporal_smoothing`| float   | 0.4       | Temporal consistency blend [0.0–1.0]             |

### Response
`application/json`

```json
{
  "run_id": "abc12345",
  "output_filename": "output_abc12345.mp4",
  "resolution": [3840, 2160],
  "frame_rate": 30.0,
  "processing_log": {
    "total_frames": 900,
    "runtime_s": 42.1,
    "mode": "balanced",
    "gpu_used": true,
    "warnings": []
  },
  "metrics": {
    "temporal_consistency_score": 0.00312,
    "mean_processing_ms_per_frame": 46.8,
    "effective_fps": 21.4,
    "hallucinated_frames": 0
  },
  "download_url": "/download/output_abc12345.mp4",
  "manifest_url": "/download/manifest_abc12345.json"
}
```

---

## POST /preview

Quick preview on first frame. Returns a PNG composite (original | enhanced | diff).

### Request
Same `file` + `mode` + `sharpen_amount`.

### Response
`image/png` — composite comparison image.

---

## GET /download/{filename}

Download an output file by name from the server temp directory.

---

## GET /health

Returns `{"status": "ok", "model_version": "1.0.0"}`.

---

## GET /metrics

Returns the metrics from the most recent `/enhance` call.

---

## Error Codes

| Code | Meaning                          |
|------|----------------------------------|
| 400  | Invalid mode, unsupported file type |
| 404  | Output file not found            |
| 500  | Internal processing error        |
