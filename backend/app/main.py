# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.image_utils import load_image_from_bytes, resize_for_processing
from app.object_detector import ObjectDetector
from app.navigation_logic import decide_navigation
from app.schemas import NavigationResponse

# We are not loading segmentation/depth for first Render deployment.
# Keep these imports disabled for now:
# from app.segmenter import SemanticSegmenter
# from app.depth_estimator import DepthEstimator


app = FastAPI(
    title="WebAR Walking Assistive Navigation Backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prototype only. Restrict later to your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model variables.
# This allows Render to open the web port quickly.
object_detector = None
segmenter = None
depth_estimator = None


@app.on_event("startup")
def startup_event():
    """
    Render needs the web server to start quickly.

    Do not load YOLO / Torch / large models here.
    Load them lazily when /analyze_frame is first called.
    """
    print("FastAPI startup complete. Models will load lazily.", flush=True)


@app.get("/")
def root():
    """
    Lightweight health check endpoint.
    Render can call this without loading YOLO.
    """
    return {
        "status": "running",
        "message": "WebAR assistive navigation backend is running."
    }


@app.post("/analyze_frame", response_model=NavigationResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Unity sends a camera frame here.

    Current Render-friendly prototype version:
    - Lazy-loads YOLO only on first request
    - Saves latest camera frame for debugging
    - Runs YOLOv8n object detection
    - Temporarily forces walkable zones to True
    - Temporarily forces depth zones to far
    - Returns navigation result to Unity
    """

    try:
        global object_detector

        print("DEBUG: analyze_frame called", flush=True)

        # 1. Lazy-load YOLO only when first needed.
        if object_detector is None:
            print("Lazy loading YOLOv8n object detector...", flush=True)
            object_detector = ObjectDetector()
            print("YOLOv8n loaded.", flush=True)

        # 2. Read image bytes from Unity upload.
        image_bytes = await file.read()

        # 3. Convert bytes to PIL image.
        image = load_image_from_bytes(image_bytes)

        # 4. Resize image for faster processing.
        image = resize_for_processing(image)

        # 5. Save latest frame for local debugging.
        # On Render this may save to temporary storage only.
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

        latest_frame_path = debug_dir / "latest_frame.jpg"
        image.save(latest_frame_path)

        print(
            f"DEBUG saved latest frame to {latest_frame_path}",
            flush=True
        )

        # 6. Run YOLO object detection.
        print("DEBUG: before YOLO detect", flush=True)

        detections = object_detector.detect(image)

        print("DEBUG: after YOLO detect", detections, flush=True)

        # 7. TEMPORARY TEST:
        # Force semantic segmentation to say all zones are walkable.
        #
        # Later, when you want real segmentation again, replace this with:
        # walkable_zones = segmenter.analyze_walkable_zones(image)
        walkable_zones = {
            "left": True,
            "center": True,
            "right": True,
        }

        # 8. TEMPORARY TEST:
        # Force depth estimation to say all zones are far.
        #
        # Later, when you want real depth again, replace this with:
        # depth_zones = depth_estimator.estimate_depth_zones(image)
        depth_zones = {
            "left": "far",
            "center": "far",
            "right": "far",
        }

        print("DEBUG walkable_zones:", walkable_zones, flush=True)
        print("DEBUG depth_zones:", depth_zones, flush=True)

        # 9. Decide final direction.
        result = decide_navigation(
            detections=detections,
            walkable_zones=walkable_zones,
            depth_zones=depth_zones,
        )

        print("DEBUG result:", result, flush=True)

        # 10. Return JSON result to Unity.
        return result

    except Exception as e:
        print("DEBUG ERROR:", str(e), flush=True)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )