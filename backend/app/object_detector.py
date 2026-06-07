# backend/app/object_detector.py

from ultralytics import YOLO

from app.config import (
    OBJECT_MODEL_NAME,
    OBJECT_CONFIDENCE_THRESHOLD,
    OBSTACLE_CLASSES,
)
from app.schemas import DetectionBox
from app.image_utils import get_zone_from_x


class ObjectDetector:
    """
    A. Object detection using backend YOLOv8n.

    Input:
        PIL image

    Output:
        List of detections with left/center/right zone.

    Debug version:
        - Prints when YOLO is called
        - Prints image size
        - Prints raw YOLO detections
        - Temporarily shows all detected classes, not only OBSTACLE_CLASSES
    """

    def __init__(self):
        print("DEBUG ObjectDetector: loading model:", OBJECT_MODEL_NAME, flush=True)
        self.model = YOLO(OBJECT_MODEL_NAME)
        print("DEBUG ObjectDetector: model loaded", flush=True)

    def detect(self, image) -> list[DetectionBox]:
        print("DEBUG ObjectDetector.detect() called", flush=True)

        width, height = image.size
        print(f"DEBUG image size: width={width}, height={height}", flush=True)

        # Lower confidence for debugging.
        # Original value may be 0.35, but 0.10 helps us see whether YOLO detects anything.
        debug_confidence = 0.10

        results = self.model.predict(
            image,
            conf=debug_confidence,
            verbose=False
        )

        print("DEBUG YOLO prediction finished", flush=True)

        detections = []

        if len(results) == 0:
            print("DEBUG YOLO returned no result objects", flush=True)
            return detections

        result = results[0]

        if result.boxes is None:
            print("DEBUG result.boxes is None", flush=True)
            return detections

        print("DEBUG number of raw YOLO boxes:", len(result.boxes), flush=True)

        class_names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = class_names[class_id]
            confidence = float(box.conf[0].item())

            print("DEBUG YOLO raw detection:", label, confidence, flush=True)

            # TEMPORARY DEBUG:
            # For testing, we do NOT filter by OBSTACLE_CLASSES.
            # This lets you see cup, bottle, cell phone, book, etc.
            #
            # Later, for safety/navigation mode, uncomment this:
            #
            # if label not in OBSTACLE_CLASSES:
            #     continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x_center = (x1 + x2) / 2.0
            zone = get_zone_from_x(x_center, width)

            detection = DetectionBox(
                label=label,
                confidence=confidence,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                zone=zone,
            )

            detections.append(detection)

        print("DEBUG final detections:", detections, flush=True)

        return detections