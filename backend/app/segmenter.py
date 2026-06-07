# backend/app/segmenter.py

import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from app.config import SEGMENTATION_MODEL_NAME


class SemanticSegmenter:
    """
    B. Semantic segmentation on backend.

    Instead of sending a full pixel mask to Unity, this class returns:
        left walkable?
        center walkable?
        right walkable?

    This keeps Unity frontend simple.
    """

    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(
            SEGMENTATION_MODEL_NAME
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            SEGMENTATION_MODEL_NAME
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

    def analyze_walkable_zones(self, image):
        """
        Returns:
            {
                "left": bool,
                "center": bool,
                "right": bool
            }
        """

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # Resize segmentation output to original image size.
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        height, width = predicted.shape

        zones = {
            "left": predicted[:, : int(width * 0.33)],
            "center": predicted[:, int(width * 0.33): int(width * 0.66)],
            "right": predicted[:, int(width * 0.66):],
        }

        result = {}

        for zone_name, zone_mask in zones.items():
            result[zone_name] = self._is_zone_walkable(zone_mask)

        return result

    def _is_zone_walkable(self, zone_mask: np.ndarray) -> bool:
        """
        Estimate if a zone is walkable.

        For a prototype, we count labels whose names contain words like:
            floor, road, sidewalk, path, ground

        This is a simple rule. Later you should tune this for your environment.
        """

        unique_ids, counts = np.unique(zone_mask, return_counts=True)

        total_pixels = zone_mask.size
        walkable_pixels = 0

        for label_id, count in zip(unique_ids, counts):
            label_name = self.id2label.get(int(label_id), "").lower()

            if (
                "floor" in label_name
                or "road" in label_name
                or "sidewalk" in label_name
                or "path" in label_name
                or "ground" in label_name
            ):
                walkable_pixels += count

        walkable_ratio = walkable_pixels / total_pixels

        # Prototype threshold.
        # If at least 20% of zone looks walkable, consider it possible.
        return walkable_ratio >= 0.20