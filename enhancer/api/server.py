"""
FastAPI REST API server
POST /enhance        — full enhancement endpoint
POST /feedback       — submit accept/reject feedback on an enhancement
POST /feedback/train — trigger feedback fine-tuning
GET  /feedback/status — feedback count and model version
GET  /health         — liveness check
GET  /metrics        — returns last run metrics
POST /preview        — quick preview on a clip/thumbnail
"""

import io
import os
import uuid
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from enhancer.pipeline.enhancer_pipeline import EnhancerPipeline, EnhancementConfig
from enhancer.utils.media_io import SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS
from enhancer.feedback.collector import FeedbackCollector
from enhancer.feedback.trainer import FeedbackTrainer

# Global instances
_pipeline: Optional[EnhancerPipeline] = None
_last_metrics: dict = {}
_feedback_collector: Optional[FeedbackCollector] = None

TEMP_DIR = Path(tempfile.gettempdir()) / "video_enhancer_outputs"
TEMP_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
FEEDBACK_DIR   = os.environ.get("FEEDBACK_DIR",   "./feedback_data")
ALLOW_FEEDBACK = os.environ.get("ALLOW_FEEDBACK",  "true").lower() == "true"


def _get_pipeline(config: EnhancementConfig) -> EnhancerPipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None or _pipeline.config.mode != config.mode:
        _pipeline = EnhancerPipeline(config=config)
        _pipeline.load_models()
    else:
        # Update config in place for non-mode settings
        _pipeline.config = config
        _pipeline.postprocessor.mode = config.mode
        _pipeline.postprocessor.max_sharpen = _pipeline.postprocessor.SHARPEN_LIMITS.get(config.mode, 0.5)
    return _pipeline


def _get_feedback_collector() -> FeedbackCollector:
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector(
            feedback_dir=FEEDBACK_DIR,
            allow_collection=ALLOW_FEEDBACK,
        )
    return _feedback_collector


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")

    app = FastAPI(
        title="Video & Image Enhancer AI",
        description="Conservative, auditable media enhancement to 4K.",
        version="1.0.0",
    )

    # CORS — allows your website's frontend to call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten to your domain in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        fc = _get_feedback_collector()
        return {
            "status": "ok",
            "model_version": "1.0.0",
            "feedback_collection": fc.allow_collection,
            "feedback_count": fc.count() if fc.allow_collection else None,
        }

    @app.get("/metrics")
    async def metrics():
        return _last_metrics or {"message": "No metrics yet. Run /enhance first."}

    @app.post("/enhance")
    async def enhance(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        target_width: int = Form(3840),
        target_height: int = Form(2160),
        mode: str = Form("balanced"),
        preserve_aspect: bool = Form(True),
        stabilize: bool = Form(True),
        aggressive: bool = Form(False),
        sharpen_amount: float = Form(0.5),
        denoise_strength: float = Form(0.5),
        temporal_smoothing: float = Form(0.4),
    ):
        global _last_metrics

        # Validate mode
        valid_modes = {"fast", "balanced", "quality", "forensics", "conservative"}
        if mode not in valid_modes:
            raise HTTPException(400, f"Invalid mode '{mode}'. Choose from: {valid_modes}")

        # Save upload to temp
        ext = Path(file.filename).suffix.lower()
        run_id = str(uuid.uuid4())[:8]
        input_path = TEMP_DIR / f"input_{run_id}{ext}"
        output_path = TEMP_DIR / f"output_{run_id}{ext}"
        manifest_path = TEMP_DIR / f"manifest_{run_id}.json"

        content = await file.read()
        with open(input_path, "wb") as f_out:
            f_out.write(content)

        config = EnhancementConfig(
            target_width=target_width,
            target_height=target_height,
            mode=mode,
            aggressive=aggressive,
            stabilize=stabilize,
            preserve_aspect=preserve_aspect,
            sharpen_amount=sharpen_amount,
            denoise_strength=denoise_strength,
            temporal_smoothing=temporal_smoothing,
        )
        pipeline = _get_pipeline(config)

        try:
            if ext in SUPPORTED_IMAGE_EXTS:
                import cv2
                img = cv2.imread(str(input_path))
                enhanced_bgr, conf_map, log = pipeline.enhance_image(img)
                cv2.imwrite(str(output_path), enhanced_bgr)
                h, w = enhanced_bgr.shape[:2]
                metrics_out = {
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "temporal_consistency_score": None,
                }
                _last_metrics = metrics_out
                return JSONResponse({
                    "run_id": run_id,
                    "output_filename": output_path.name,
                    "resolution": [w, h],
                    "frame_rate": None,
                    "processing_log": log,
                    "metrics": metrics_out,
                    "hallucinated_fraction": log.get("hallucinated_fraction"),
                    "download_url": f"/download/{output_path.name}",
                    "manifest_url": f"/download/{manifest_path.name}",
                })

            elif ext in SUPPORTED_VIDEO_EXTS:
                frames_collected = []

                def progress(idx, total, fps_est):
                    pass  # Could emit SSE events here

                plog = pipeline.enhance_video(str(input_path), str(output_path), callback=progress)
                pipeline.generate_manifest(str(manifest_path), plog)

                metrics_out = plog.metrics
                metrics_out["hallucinated_frames"] = len(plog.hallucinated_frames)
                _last_metrics = metrics_out

                return JSONResponse({
                    "run_id": run_id,
                    "output_filename": output_path.name,
                    "resolution": plog.target_resolution,
                    "frame_rate": plog.fps,
                    "processing_log": {
                        "total_frames": plog.total_frames,
                        "runtime_s": plog.total_runtime_s,
                        "mode": plog.mode,
                        "gpu_used": plog.gpu_used,
                        "warnings": plog.warnings,
                    },
                    "metrics": metrics_out,
                    "download_url": f"/download/{output_path.name}",
                    "manifest_url": f"/download/{manifest_path.name}",
                })
            else:
                raise HTTPException(400, f"Unsupported file type: {ext}")

        finally:
            # Clean up input after processing
            background_tasks.add_task(lambda: input_path.unlink(missing_ok=True))

    @app.post("/preview")
    async def preview(
        file: UploadFile = File(...),
        mode: str = Form("balanced"),
        sharpen_amount: float = Form(0.5),
    ):
        """
        Quick preview: enhances first frame or thumbnail-sized crop.
        Returns a comparison PNG (original | enhanced).
        """
        import cv2
        ext = Path(file.filename).suffix.lower()
        run_id = str(uuid.uuid4())[:8]
        input_path = TEMP_DIR / f"preview_in_{run_id}{ext}"
        output_path = TEMP_DIR / f"preview_out_{run_id}.png"

        content = await file.read()
        with open(input_path, "wb") as f_out:
            f_out.write(content)

        config = EnhancementConfig(mode=mode, sharpen_amount=sharpen_amount)
        pipeline = _get_pipeline(config)

        if ext in SUPPORTED_VIDEO_EXTS:
            cap = cv2.VideoCapture(str(input_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(400, "Cannot read video frame")
        else:
            frame = cv2.imread(str(input_path))

        # Downscale for fast preview
        h, w = frame.shape[:2]
        preview_w = min(w, 1280)
        preview_h = int(h * preview_w / w)
        frame_small = cv2.resize(frame, (preview_w, preview_h))

        enhanced_bgr, conf_map, log = pipeline.enhance_image(frame_small)

        from enhancer.utils.visual_diff import generate_visual_diff
        _, _, composite = generate_visual_diff(frame_small, enhanced_bgr, str(output_path))
        input_path.unlink(missing_ok=True)

        return FileResponse(str(output_path), media_type="image/png", filename=f"preview_{run_id}.png")

    @app.get("/download/{filename}")
    async def download(filename: str):
        """Download an output file by name."""
        safe_name = Path(filename).name
        file_path = TEMP_DIR / safe_name
        if not file_path.exists():
            raise HTTPException(404, "File not found")
        return FileResponse(str(file_path), filename=safe_name)

    # ------------------------------------------------------------------
    # Feedback endpoints
    # ------------------------------------------------------------------

    @app.post("/feedback")
    async def submit_feedback(
        run_id: str = Form(...),
        verdict: str = Form(...),           # "accept" | "reject" | "partial"
        notes: str = Form(""),
        mode: str = Form("balanced"),
        mask: Optional[UploadFile] = File(None),  # optional region mask PNG
    ):
        """
        Submit feedback on an enhancement result.
        Requires the run_id returned by /enhance.
        """
        fc = _get_feedback_collector()
        if not fc.allow_collection:
            return JSONResponse({"message": "Feedback collection is disabled (opt-out)."})

        valid_verdicts = {"accept", "reject", "partial"}
        if verdict not in valid_verdicts:
            raise HTTPException(400, f"verdict must be one of {valid_verdicts}")

        # Locate the input and output files from this run
        input_candidates  = list(TEMP_DIR.glob(f"input_{run_id}*"))
        output_candidates = list(TEMP_DIR.glob(f"output_{run_id}*"))

        if not input_candidates or not output_candidates:
            raise HTTPException(404, f"No files found for run_id '{run_id}'. "
                                     f"Files may have been cleaned up.")

        import cv2
        input_bgr  = cv2.imread(str(input_candidates[0]))
        output_bgr = cv2.imread(str(output_candidates[0]))

        if input_bgr is None or output_bgr is None:
            raise HTTPException(400, "Could not read run files (may be a video — image feedback only).")

        # Load optional mask
        mask_arr = None
        if mask and verdict == "partial":
            mask_bytes = await mask.read()
            mask_arr_np = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask_arr = cv2.imdecode(mask_arr_np, cv2.IMREAD_GRAYSCALE)

        fid = fc.submit(
            input_bgr=input_bgr,
            output_bgr=output_bgr,
            verdict=verdict,
            region_mask=mask_arr,
            mode=mode,
            notes=notes,
        )
        counts = fc.count()
        return JSONResponse({
            "feedback_id": fid,
            "verdict": verdict,
            "total_feedback": counts["total"],
            "accept_count": counts["accept"],
            "reject_count": counts["reject"],
            "message": f"Feedback recorded. {counts['total']} total entries."
        })

    @app.get("/feedback/status")
    async def feedback_status():
        """Returns feedback count, model version, and training readiness."""
        fc = _get_feedback_collector()
        if not fc.allow_collection:
            return {"feedback_enabled": False}

        counts = fc.count()
        version_file = os.path.join(CHECKPOINT_DIR, "feedback_version.json")
        version_info = {}
        if os.path.exists(version_file):
            with open(version_file) as f:
                version_info = json.load(f)

        return {
            "feedback_enabled": True,
            "counts": counts,
            "ready_to_train": counts["accept"] >= 10,
            "min_pairs_needed": max(0, 10 - counts["accept"]),
            "model_version": version_info.get("version", 0),
            "last_trained": version_info.get("latest", {}).get("timestamp"),
        }

    @app.post("/feedback/train")
    async def trigger_training(background_tasks: BackgroundTasks):
        """
        Trigger a feedback fine-tuning run in the background.
        Returns immediately; training runs async.
        """
        fc = _get_feedback_collector()
        if not fc.allow_collection:
            raise HTTPException(400, "Feedback collection is disabled.")

        counts = fc.count()
        if counts["accept"] < 10:
            raise HTTPException(400,
                f"Not enough accepted feedback. Have {counts['accept']}, need 10.")

        pairs = fc.get_training_pairs()

        def run_training():
            pipeline = _pipeline
            if pipeline is None or not pipeline._models_loaded:
                return
            trainer = FeedbackTrainer(
                model=pipeline._image_sr,
                checkpoint_dir=CHECKPOINT_DIR,
                device=pipeline.device,
            )
            result = trainer.train(pairs)
            print(f"[API] Feedback training result: {result}")

        background_tasks.add_task(run_training)
        return JSONResponse({
            "message": "Training started in background.",
            "pairs_used": len(pairs),
            "note": "Check /feedback/status for model version update when complete."
        })

    @app.post("/feedback/optout")
    async def opt_out():
        """Permanently opt out of feedback collection and delete all stored data."""
        fc = _get_feedback_collector()
        fc.opt_out()
        return JSONResponse({"message": "Opted out. All feedback data deleted."})

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI/uvicorn not installed.")
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
