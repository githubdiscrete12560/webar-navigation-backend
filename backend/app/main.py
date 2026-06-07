# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.image_utils import load_image_from_bytes, resize_for_processing
from app.object_detector import ObjectDetector
from app.segmenter import SemanticSegmenter
from app.depth_estimator import DepthEstimator
from app.navigation_logic import decide_navigation
from app.schemas import NavigationResponse

app = FastAPI(
    title="WebAR Walking Assistive Navigation Backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prototype only. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

object_detector = None
segmenter = None
depth_estimator = None


@app.on_event("startup")
def startup_event():
    global object_detector, segmenter, depth_estimator

    print("Loading YOLOv8n object detector...")
    object_detector = ObjectDetector()

    print("Loading semantic segmentation model...")
    segmenter = SemanticSegmenter()

    print("Loading monocular depth model...")
    depth_estimator = DepthEstimator()

    print("All backend models loaded.")


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "WebAR assistive navigation backend is running."
    }


@app.post("/analyze_frame", response_model=NavigationResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Unity sends a camera frame here.

    Current debug/testing version:
    - Saves the latest camera frame to backend/debug_frames/latest_frame.jpg
    - Runs YOLOv8n object detection
    - Temporarily forces walkable zones to True
    - Temporarily forces depth zones to far
    - Returns simplified navigation result to Unity
    """

    try:
        print("DEBUG false logic: analyze_frame called", flush=True)

        # 1. Read image bytes from Unity upload
        image_bytes = await file.read()

        # 2. Convert bytes to PIL image
        image = load_image_from_bytes(image_bytes)

        # 3. Resize image for faster processing
        image = resize_for_processing(image)

        # 4. Save latest frame for debugging
        # Open this file to see exactly what YOLO receives.
        from pathlib import Path

        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

        latest_frame_path = debug_dir / "latest_frame.jpg"
        image.save(latest_frame_path)

        print(
            f"DEBUG saved latest frame to {latest_frame_path}",
            flush=True
        )

        # 5. Run YOLO object detection
        print("DEBUG: before YOLO detect", flush=True)

        detections = object_detector.detect(image)

        print("DEBUG: after YOLO detect", detections, flush=True)

        # 6. TEMPORARY TEST:
        # Force semantic segmentation to say all zones are walkable.
        # Later, restore:
        # walkable_zones = segmenter.analyze_walkable_zones(image)
        walkable_zones = {
            "left": True,
            "center": True,
            "right": True,
        }

        # 7. TEMPORARY TEST:
        # Force depth estimation to say all zones are far.
        # Later, restore:
        # depth_zones = depth_estimator.estimate_depth_zones(image)
        depth_zones = {
            "left": "far",
            "center": "far",
            "right": "far",
        }

        print("DEBUG walkable_zones:", walkable_zones, flush=True)
        print("DEBUG depth_zones:", depth_zones, flush=True)

        # 8. Decide final direction
        result = decide_navigation(
            detections=detections,
            walkable_zones=walkable_zones,
            depth_zones=depth_zones,
        )

        print("DEBUG result:", result, flush=True)

        # 9. Return JSON result to Unity
        return result

    except Exception as e:
        print("DEBUG ERROR:", str(e), flush=True)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )