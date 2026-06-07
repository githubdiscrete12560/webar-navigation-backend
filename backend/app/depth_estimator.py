# backend/app/depth_estimator.py

import torch
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation

from app.config import DEPTH_MODEL_NAME


class DepthEstimator:
    """
    C. Backend monocular depth estimation.

    This gives relative depth, not exact meters.

    Output:
        {
            "left": "near" | "medium" | "far",
            "center": "near" | "medium" | "far",
            "right": "near" | "medium" | "far"
        }
    """

    def __init__(self):
        self.processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
        self.model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
        self.model.eval()

    def estimate_depth_zones(self, image):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        # Resize depth map to original image size.
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth = depth.squeeze().cpu().numpy()

        # Normalize to 0-1.
        # In many monocular depth models, larger value often means closer,
        # but always test with your chosen model.
        depth_min = depth.min()
        depth_max = depth.max()

        normalized_depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        height, width = normalized_depth.shape

        left_zone = normalized_depth[:, : int(width * 0.33)]
        center_zone = normalized_depth[:, int(width * 0.33): int(width * 0.66)]
        right_zone = normalized_depth[:, int(width * 0.66):]

        return {
            "left": self._classify_zone_depth(left_zone),
            "center": self._classify_zone_depth(center_zone),
            "right": self._classify_zone_depth(right_zone),
        }

    def _classify_zone_depth(self, zone_depth: np.ndarray) -> str:
        """
        Classify zone as near, medium, or far.

        We use a high percentile to focus on close obstacles.
        """
        close_score = float(np.percentile(zone_depth, 85))

        if close_score >= 0.65:
            return "near"

        if close_score >= 0.45:
            return "medium"

        return "far"