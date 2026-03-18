from __future__ import annotations

from typing import Dict, List


BEST_PROMPT = (
    "First-person camera moving forward along an outdoor park walkway. "
    "Photorealistic. Stable exposure and white balance. Consistent perspective and geometry, "
    "level horizon. Sharp, stable textures on pavement, grass, and trees. No flicker, no warping, no artifacts."
)

BEST_NEGATIVE_PROMPT = (
    "blurry, flickering, warping, wobble, rolling shutter artifacts, ghosting, double edges, "
    "inconsistent geometry, wrong perspective, texture swimming, repeating patterns, oversharpening halos, "
    "heavy motion blur, text, watermark, jpeg artifacts, excessive noise, color shift, low quality, "
    "crowds, many people, fast moving objects"
)

SIDE_STRUCTURE_PRIORITY = [
    "grass",
    "trees",
    "shrubs",
    "walls",
    "columns",
    "fence",
    "buildings",
    "crops",
    "soil_edges",
    "hedges",
]  # type: List[str]


SCENE_PHRASE_MAP = {
    "park_walkway": "outdoor park walkway",
    "campus_path": "outdoor campus path",
    "orchard_row": "row path in an orchard",
    "field_path": "field path",
    "road_edge": "outdoor road edge",
    "corridor": "indoor corridor",
    "indoor_walkway": "indoor walkway",
    "unknown": "outdoor walkway",
}  # type: Dict[str, str]

SURFACE_PHRASE_MAP = {
    "pavement": "pavement",
    "concrete": "concrete pavement",
    "brick": "brick pavement",
    "dirt": "dirt path",
    "mixed": "path surface",
    "unknown": "path surface",
}  # type: Dict[str, str]

SIDE_STRUCTURE_PHRASE_MAP = {
    "grass": "grass",
    "trees": "trees",
    "shrubs": "shrubs",
    "walls": "walls",
    "columns": "columns",
    "fence": "fence",
    "buildings": "buildings",
    "crops": "crops",
    "soil_edges": "soil edges",
    "hedges": "hedges",
}  # type: Dict[str, str]

BASE_NEGATIVE_ITEMS = [
    "blurry",
    "flickering",
    "warping",
    "wobble",
    "rolling shutter artifacts",
    "ghosting",
    "double edges",
    "inconsistent geometry",
    "wrong perspective",
    "texture swimming",
    "repeating patterns",
    "oversharpening halos",
    "heavy motion blur",
    "text",
    "watermark",
    "jpeg artifacts",
    "excessive noise",
    "color shift",
    "low quality",
    "crowds",
    "many people",
    "fast moving objects",
]  # type: List[str]

DYNAMIC_NEGATIVE_ITEMS = {
    "low": [],
    "people": ["pedestrians"],
    "vehicles": ["cars", "vehicles", "cyclists"],
    "animals": ["animals", "moving creatures"],
    "mixed": ["vehicles", "animals"],
}  # type: Dict[str, List[str]]

REPETITION_NEGATIVE_ITEMS = {
    "low": [],
    "medium": [],
    "high": ["duplicate structures", "repeated textures"],
}  # type: Dict[str, List[str]]

FORBIDDEN_POSITIVE_TOKENS = [
    "person",
    "people",
    "car",
    "bike",
    "animal",
    "bench",
    "lamp",
    "sign",
    "background",
    "right side",
    "left side",
    "near",
    "far",
    "blue line",
]  # type: List[str]
