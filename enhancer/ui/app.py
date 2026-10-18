"""
Gradio Web UI
Provides:
- File upload (image or video)
- Mode/preset selection
- Advanced sliders
- Preview button
- Full enhance button
- Output display: enhanced media + confidence overlay + diff heatmap
- Processing log display
"""

import os
import tempfile
import json
import numpy as np
import cv2

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from enhancer.pipeline.enhancer_pipeline import EnhancerPipeline, EnhancementConfig
from enhancer.utils.confidence_map import overlay_hallucination_mask
from enhancer.utils.visual_diff import generate_visual_diff


_pipeline: EnhancerPipeline = None


def _get_pipeline(mode: str, sharpen: float, denoise: float, temporal: float) -> EnhancerPipeline:
    global _pipeline
    config = EnhancementConfig(
        mode=mode,
        sharpen_amount=sharpen,
        denoise_strength=denoise,
        temporal_smoothing=temporal,
    )
    if _pipeline is None:
        _pipeline = EnhancerPipeline(config=config)
        _pipeline.load_models()
    else:
        _pipeline.config = config
    return _pipeline


def enhance_image_ui(
    file_path: str,
    mode: str,
    sharpen_amount: float,
    denoise_strength: float,
    temporal_smoothing: float,
    aggressive: bool,
) -> tuple:
    """
    Called by Gradio for image enhancement.
    Returns (enhanced_rgb, confidence_overlay_rgb, diff_heatmap_rgb, log_text).
    """
    pipeline = _get_pipeline(mode, sharpen_amount, denoise_strength, temporal_smoothing)
    pipeline.config.aggressive = aggressive

    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        return None, None, None, "Error: Cannot read image file."

    enhanced_bgr, conf_map, log = pipeline.enhance_image(img_bgr)

    # Confidence overlay
    conf_overlay = overlay_hallucination_mask(enhanced_bgr, conf_map)
    conf_overlay_rgb = cv2.cvtColor(conf_overlay, cv2.COLOR_BGR2RGB)

    # Visual diff
    _, diff_hm, _ = generate_visual_diff(img_bgr, enhanced_bgr)
    diff_rgb = cv2.cvtColor(diff_hm, cv2.COLOR_BGR2RGB)

    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    log_text = json.dumps(log, indent=2)
    return enhanced_rgb, conf_overlay_rgb, diff_rgb, log_text


def enhance_video_ui(
    file_path: str,
    mode: str,
    sharpen_amount: float,
    denoise_strength: float,
    temporal_smoothing: float,
    aggressive: bool,
    progress=None,
) -> tuple:
    """
    Called by Gradio for video enhancement.
    Returns (output_video_path, log_text).
    """
    pipeline = _get_pipeline(mode, sharpen_amount, denoise_strength, temporal_smoothing)
    pipeline.config.aggressive = aggressive

    out_path = os.path.join(tempfile.gettempdir(), "enhanced_output.mp4")

    def _progress(idx, total, fps_est):
        if progress:
            progress(idx / max(total, 1), desc=f"Frame {idx}/{total} ({fps_est:.1f} fps)")

    plog = pipeline.enhance_video(file_path, out_path, callback=_progress)
    log_text = json.dumps({
        "mode": plog.mode,
        "total_frames": plog.total_frames,
        "runtime_s": plog.total_runtime_s,
        "metrics": plog.metrics,
        "hallucinated_frames_count": len(plog.hallucinated_frames),
        "warnings": plog.warnings,
    }, indent=2)

    return out_path, log_text


def build_ui() -> "gr.Blocks":
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not installed. pip install gradio")

    with gr.Blocks(title="Video & Image Enhancer AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Video & Image Enhancer AI")
        gr.Markdown(
            "Conservative, auditable enhancement to 4K. "
            "Supports JPEG, PNG, TIFF, MP4, MOV, AVI. "
            "Confidence maps and processing logs included."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                mode_select = gr.Radio(
                    choices=["conservative", "balanced", "quality", "fast", "forensics"],
                    value="balanced",
                    label="Enhancement Mode",
                )
                aggressive_cb = gr.Checkbox(label="Aggressive mode (explicit opt-in)", value=False)

                gr.Markdown("#### Advanced Controls")
                sharpen_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Sharpen Amount")
                denoise_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Denoise Strength")
                temporal_sl = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Temporal Smoothing")

                gr.Markdown("---")
                gr.Markdown("**Mode descriptions:**")
                gr.Markdown(
                    "- **conservative**: minimal changes, highest fidelity\n"
                    "- **balanced**: default, good detail vs. artifact tradeoff\n"
                    "- **quality**: slower, higher-quality SR + deblur\n"
                    "- **fast**: speed-optimized, reduced model depth\n"
                    "- **forensics**: no hallucination, full audit logs"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Image Enhancement")
                with gr.Tab("Image"):
                    img_input = gr.Image(type="filepath", label="Input Image")
                    with gr.Row():
                        img_preview_btn = gr.Button("Preview (fast)", variant="secondary")
                        img_enhance_btn = gr.Button("Enhance", variant="primary")

                    img_output = gr.Image(type="numpy", label="Enhanced Output")
                    with gr.Row():
                        conf_overlay_out = gr.Image(type="numpy", label="Confidence Overlay (red = uncertain)")
                        diff_out = gr.Image(type="numpy", label="Pixel Difference Heatmap")
                    img_log = gr.Textbox(label="Processing Log", lines=15, interactive=False)

                with gr.Tab("Video"):
                    vid_input = gr.Video(label="Input Video")
                    vid_enhance_btn = gr.Button("Enhance Video", variant="primary")
                    vid_output = gr.Video(label="Enhanced Output")
                    vid_log = gr.Textbox(label="Processing Log", lines=15, interactive=False)

        # Wire up image buttons
        common_inputs = [mode_select, sharpen_sl, denoise_sl, temporal_sl, aggressive_cb]

        img_enhance_btn.click(
            fn=enhance_image_ui,
            inputs=[img_input] + common_inputs,
            outputs=[img_output, conf_overlay_out, diff_out, img_log],
        )

        img_preview_btn.click(
            fn=lambda fp, m, sh, dn, ts, ag: enhance_image_ui(fp, m, sh * 0.5, dn * 0.5, ts, ag),
            inputs=[img_input] + common_inputs,
            outputs=[img_output, conf_overlay_out, diff_out, img_log],
        )

        vid_enhance_btn.click(
            fn=enhance_video_ui,
            inputs=[vid_input] + common_inputs,
            outputs=[vid_output, vid_log],
        )

    return demo


def launch(share: bool = False, port: int = 7860):
    demo = build_ui()
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch()
