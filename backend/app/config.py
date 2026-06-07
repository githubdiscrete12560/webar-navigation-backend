# backend/app/config.py

OBJECT_MODEL_NAME = "yolov8n.pt"

# Segmentation model.
# This is a semantic segmentation model, useful for identifying regions such as floor, wall, road, etc.
SEGMENTATION_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Depth model.
# MiDaS small is a practical starting point for relative monocular depth.
DEPTH_MODEL_NAME = "Intel/dpt-hybrid-midas"

# Image processing size.
MAX_IMAGE_WIDTH = 640

# YOLO confidence threshold.
OBJECT_CONFIDENCE_THRESHOLD = 0.35

# Zone thresholds.
LEFT_ZONE_END_RATIO = 0.33
RIGHT_ZONE_START_RATIO = 0.66

# Safety thresholds.
NEAR_DEPTH_THRESHOLD = 0.65
MEDIUM_DEPTH_THRESHOLD = 0.45

# Labels that should be treated as walking obstacles.
OBSTACLE_CLASSES = {
    "person",
    "chair",
    "bench",
    "couch",
    "bed",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "dog",
    "cat",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
}

# ADE20K-like labels that may indicate walkable areas.
# Label availability depends on the segmentation model.
WALKABLE_LABEL_HINTS = {
    "floor",
    "road",
    "sidewalk",
    "path",
    "ground",
}

# Labels that should be treated as blocked/non-walkable.
BLOCKING_LABEL_HINTS = {
    "wall",
    "table",
    "chair",
    "sofa",
    "bed",
    "stairs",
    "person",
    "door",
    "cabinet",
}