"""
Feedback Collector
Stores explicit user feedback (accept/reject/region) as labeled pairs
on disk. Only explicit labeled feedback is ever collected — no silent
data collection. Users can opt out entirely via allow_collection=False.

Feedback format (per entry in feedback_log.json):
{
  "id": "<uuid>",
  "timestamp": "ISO8601",
  "input_path": "...",
  "output_path": "...",
  "verdict": "accept" | "reject" | "partial",
  "region_mask_path": "...",   # optional: path to binary PNG mask of accepted region
  "mode": "balanced",
  "opted_in": true
}
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from typing import Optional
import numpy as np
import cv2


class FeedbackCollector:
    """
    Collects and stores user feedback on enhancement results.
    All storage is local — nothing is sent anywhere.
    """

    def __init__(self, feedback_dir: str = "./feedback_data", allow_collection: bool = True):
        """
        Args:
            feedback_dir: local directory to store feedback pairs
            allow_collection: if False, all submit() calls are no-ops (opt-out)
        """
        self.feedback_dir = feedback_dir
        self.allow_collection = allow_collection

        if allow_collection:
            os.makedirs(os.path.join(feedback_dir, "inputs"), exist_ok=True)
            os.makedirs(os.path.join(feedback_dir, "outputs"), exist_ok=True)
            os.makedirs(os.path.join(feedback_dir, "masks"), exist_ok=True)

        self._log_path = os.path.join(feedback_dir, "feedback_log.json")
        self._log: list = []
        if allow_collection and os.path.exists(self._log_path):
            with open(self._log_path) as f:
                self._log = json.load(f)

    def submit(
        self,
        input_bgr: np.ndarray,
        output_bgr: np.ndarray,
        verdict: str,                          # "accept" | "reject" | "partial"
        region_mask: Optional[np.ndarray] = None,  # uint8 binary mask, 255=accepted
        mode: str = "balanced",
        notes: str = "",
    ) -> Optional[str]:
        """
        Save a feedback pair to disk.

        Args:
            input_bgr:   original LR input image
            output_bgr:  enhanced output image
            verdict:     "accept" (good output), "reject" (bad output),
                         "partial" (some regions good, use mask)
            region_mask: optional binary mask (255 = good region)
            mode:        enhancement mode used
            notes:       optional free-text note
        Returns:
            feedback_id if collected, None if opted out
        """
        if not self.allow_collection:
            return None

        valid_verdicts = {"accept", "reject", "partial"}
        if verdict not in valid_verdicts:
            raise ValueError(f"verdict must be one of {valid_verdicts}")

        fid = str(uuid.uuid4())[:12]
        ts = datetime.utcnow().isoformat()

        in_path  = os.path.join(self.feedback_dir, "inputs",  f"{fid}_input.png")
        out_path = os.path.join(self.feedback_dir, "outputs", f"{fid}_output.png")
        mask_path = None

        cv2.imwrite(in_path, input_bgr)
        cv2.imwrite(out_path, output_bgr)

        if region_mask is not None and verdict == "partial":
            mask_path = os.path.join(self.feedback_dir, "masks", f"{fid}_mask.png")
            cv2.imwrite(mask_path, region_mask)

        entry = {
            "id": fid,
            "timestamp": ts,
            "input_path": in_path,
            "output_path": out_path,
            "region_mask_path": mask_path,
            "verdict": verdict,
            "mode": mode,
            "notes": notes,
            "opted_in": True,
        }
        self._log.append(entry)
        self._save_log()
        return fid

    def get_training_pairs(self, verdict_filter: Optional[str] = None):
        """
        Returns list of (input_path, output_path, mask_path, verdict) tuples
        suitable for feeding to FeedbackTrainer.

        Args:
            verdict_filter: if set, only return entries with this verdict
        """
        pairs = []
        for entry in self._log:
            if verdict_filter and entry["verdict"] != verdict_filter:
                continue
            if os.path.exists(entry["input_path"]) and os.path.exists(entry["output_path"]):
                pairs.append((
                    entry["input_path"],
                    entry["output_path"],
                    entry.get("region_mask_path"),
                    entry["verdict"],
                ))
        return pairs

    def count(self) -> dict:
        """Returns counts by verdict."""
        counts = {"accept": 0, "reject": 0, "partial": 0, "total": len(self._log)}
        for e in self._log:
            counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
        return counts

    def opt_out(self) -> None:
        """
        Permanently opt out — clears all stored feedback and disables collection.
        This cannot be undone.
        """
        self.allow_collection = False
        if os.path.exists(self.feedback_dir):
            shutil.rmtree(self.feedback_dir)
        self._log = []
        print("[FeedbackCollector] Opted out. All feedback data deleted.")

    def _save_log(self) -> None:
        with open(self._log_path, "w") as f:
            json.dump(self._log, f, indent=2)
