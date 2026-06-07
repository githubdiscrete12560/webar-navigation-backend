# backend/app/image_utils.py

from PIL import Image
import numpy as np
import cv2
from app.config import MAX_IMAGE_WIDTH


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load uploaded image bytes as RGB PIL image.
    """
    import io

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def resize_for_processing(image: Image.Image) -> Image.Image:
    """
    Resize large images to reduce backend latency.
    """
    width, height = image.size

    if width <= MAX_IMAGE_WIDTH:
        return image

    new_height = int(height * (MAX_IMAGE_WIDTH / width))
    return image.resize((MAX_IMAGE_WIDTH, new_height))


def pil_to_cv2(image: Image.Image):
    """
    Convert PIL RGB image to OpenCV BGR image.
    """
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def get_zone_from_x(x_center: float, image_width: int) -> str:
    """
    Convert x coordinate into left / center / right zone.
    """
    left_end = image_width * 0.33
    right_start = image_width * 0.66

    if x_center < left_end:
        return "left"

    if x_center > right_start:
        return "right"

    return "center"