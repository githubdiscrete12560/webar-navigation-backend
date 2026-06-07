# backend/app/navigation_logic.py

from app.schemas import (
    DetectionBox,
    ZoneResult,
    DepthZoneResult,
    NavigationResponse,
)


def zone_blocked_by_objects(detections: list[DetectionBox]) -> dict:
    """
    Convert YOLO detections into left/center/right blocked zones.
    """

    blocked = {
        "left": False,
        "center": False,
        "right": False,
    }

    for det in detections:
        blocked[det.zone] = True

    return blocked


def decide_navigation(
    detections: list[DetectionBox],
    walkable_zones: dict,
    depth_zones: dict,
) -> NavigationResponse:
    """
    Final safety logic.

    Backend decides only:
        forward / left / right / stop

    Unity frontend only displays the result.
    """

    object_blocked = zone_blocked_by_objects(detections)

    detected_objects = sorted(list(set([d.label for d in detections])))

    max_confidence = 0.0
    if detections:
        max_confidence = max([d.confidence for d in detections])

    # Safety conditions for center path.
    center_has_object = object_blocked["center"]
    center_not_walkable = not walkable_zones["center"]
    center_near = depth_zones["center"] == "near"

    left_possible = (
        not object_blocked["left"]
        and walkable_zones["left"]
        and depth_zones["left"] != "near"
    )

    center_possible = (
        not center_has_object
        and walkable_zones["center"]
        and depth_zones["center"] != "near"
    )

    right_possible = (
        not object_blocked["right"]
        and walkable_zones["right"]
        and depth_zones["right"] != "near"
    )

    if center_possible:
        direction = "forward"
        warning = "path_clear"
        explanation = "Center zone appears walkable and not near-blocked."

    elif left_possible:
        direction = "left"
        warning = "center_blocked_choose_left"
        explanation = "Center zone is blocked or unsafe. Left zone appears safer."

    elif right_possible:
        direction = "right"
        warning = "center_blocked_choose_right"
        explanation = "Center zone is blocked or unsafe. Right zone appears safer."

    else:
        direction = "stop"
        warning = "no_safe_zone"
        explanation = "No zone is confident enough. Stop for safety."

    # Extra safety rule.
    if center_near and center_has_object:
        direction = "stop"
        warning = "near_object_ahead"
        explanation = "Object detection and depth both indicate a near obstacle ahead."

    # If semantic segmentation says center is not walkable, be conservative.
    if center_not_walkable and not left_possible and not right_possible:
        direction = "stop"
        warning = "center_not_walkable"
        explanation = "Segmentation does not find a safe walkable center area."

    return NavigationResponse(
        direction=direction,
        warning=warning,
        confidence=float(max_confidence),
        detected_objects=detected_objects,
        detections=detections,

        object_blocked=ZoneResult(
            left=object_blocked["left"],
            center=object_blocked["center"],
            right=object_blocked["right"],
        ),

        walkable_area=ZoneResult(
            left=walkable_zones["left"],
            center=walkable_zones["center"],
            right=walkable_zones["right"],
        ),

        depth_level=DepthZoneResult(
            left=depth_zones["left"],
            center=depth_zones["center"],
            right=depth_zones["right"],
        ),

        explanation=explanation,
    )