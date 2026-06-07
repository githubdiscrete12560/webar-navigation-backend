# backend/app/schemas.py

from pydantic import BaseModel
from typing import List


class DetectionBox(BaseModel):
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    zone: str


class ZoneResult(BaseModel):
    left: bool
    center: bool
    right: bool


class DepthZoneResult(BaseModel):
    left: str
    center: str
    right: str


class NavigationResponse(BaseModel):
    direction: str
    warning: str
    confidence: float

    detected_objects: List[str]
    detections: List[DetectionBox]

    object_blocked: ZoneResult
    walkable_area: ZoneResult
    depth_level: DepthZoneResult

    explanation: str