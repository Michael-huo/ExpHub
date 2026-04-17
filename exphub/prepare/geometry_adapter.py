from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GeometryAdaptationResult:
    frames: List[Any]
    original_resolution: Dict[str, int]
    normalized_resolution: Dict[str, int]
    original_intrinsics: Dict[str, object]
    normalized_intrinsics: Dict[str, object]
    transform_meta: Dict[str, object]


def _intrinsics_dict(intrinsics):
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "dist": [float(item) for item in list(intrinsics.get("dist") or [])],
    }


def compute_max_legal_resolution(width, height, multiple=32):
    width_i = int(width)
    height_i = int(height)
    multiple_i = int(multiple)
    if width_i <= 0 or height_i <= 0:
        raise ValueError("image resolution must be positive, got {}x{}".format(width_i, height_i))
    if multiple_i <= 0:
        raise ValueError("multiple must be > 0, got {}".format(multiple))

    normalized_width = int(width_i // multiple_i * multiple_i)
    normalized_height = int(height_i // multiple_i * multiple_i)
    if normalized_width <= 0 or normalized_height <= 0:
        raise ValueError(
            "image resolution {}x{} is smaller than legal multiple {}".format(width_i, height_i, multiple_i)
        )
    return normalized_width, normalized_height


def build_center_crop_transform(width, height, normalized_width, normalized_height):
    width_i = int(width)
    height_i = int(height)
    normalized_width_i = int(normalized_width)
    normalized_height_i = int(normalized_height)
    if normalized_width_i > width_i or normalized_height_i > height_i:
        raise ValueError(
            "normalized resolution cannot exceed original resolution: {}x{} > {}x{}".format(
                normalized_width_i,
                normalized_height_i,
                width_i,
                height_i,
            )
        )

    crop_left = int((width_i - normalized_width_i) // 2)
    crop_top = int((height_i - normalized_height_i) // 2)
    mode = "crop" if (normalized_width_i != width_i or normalized_height_i != height_i) else "crop"
    return {
        "mode": mode,
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(normalized_width_i),
        "crop_height": int(normalized_height_i),
        "scale_x": 1.0,
        "scale_y": 1.0,
    }


def adapt_intrinsics_for_transform(intrinsics, transform_meta):
    source = _intrinsics_dict(intrinsics)
    crop_left = int(transform_meta["crop_left"])
    crop_top = int(transform_meta["crop_top"])
    scale_x = float(transform_meta["scale_x"])
    scale_y = float(transform_meta["scale_y"])
    return {
        "fx": float(source["fx"]) * scale_x,
        "fy": float(source["fy"]) * scale_y,
        "cx": (float(source["cx"]) - float(crop_left)) * scale_x,
        "cy": (float(source["cy"]) - float(crop_top)) * scale_y,
        "dist": list(source["dist"]),
    }


def adapt_frame_geometry(frame, transform_meta):
    crop_left = int(transform_meta["crop_left"])
    crop_top = int(transform_meta["crop_top"])
    crop_width = int(transform_meta["crop_width"])
    crop_height = int(transform_meta["crop_height"])
    scale_x = float(transform_meta["scale_x"])
    scale_y = float(transform_meta["scale_y"])

    cropped = frame[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
    if abs(scale_x - 1.0) < 1e-12 and abs(scale_y - 1.0) < 1e-12:
        return cropped.copy()

    import cv2

    target_width = max(1, int(round(float(crop_width) * scale_x)))
    target_height = max(1, int(round(float(crop_height) * scale_y)))
    interpolation = cv2.INTER_AREA if scale_x < 1.0 or scale_y < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)


def _validate_frame_resolution(frame, expected_width, expected_height):
    actual_height, actual_width = frame.shape[:2]
    if int(actual_width) != int(expected_width) or int(actual_height) != int(expected_height):
        raise ValueError(
            "frame resolution {}x{} does not match datasets.json image_size {}x{}".format(
                int(actual_width),
                int(actual_height),
                int(expected_width),
                int(expected_height),
            )
        )


def adapt_frames_and_intrinsics(frames, intrinsics, image_size=None, multiple=32):
    frame_list = list(frames or [])
    if not frame_list:
        raise ValueError("cannot adapt geometry: no frames were provided")

    if image_size is None:
        first_height, first_width = frame_list[0].shape[:2]
        original_width, original_height = int(first_width), int(first_height)
    else:
        original_width, original_height = int(image_size[0]), int(image_size[1])
        _validate_frame_resolution(frame_list[0], original_width, original_height)

    normalized_width, normalized_height = compute_max_legal_resolution(original_width, original_height, multiple)
    transform_meta = build_center_crop_transform(
        original_width,
        original_height,
        normalized_width,
        normalized_height,
    )
    normalized_intrinsics = adapt_intrinsics_for_transform(intrinsics, transform_meta)
    adapted_frames = [adapt_frame_geometry(frame, transform_meta) for frame in frame_list]

    return GeometryAdaptationResult(
        frames=adapted_frames,
        original_resolution={"width": int(original_width), "height": int(original_height)},
        normalized_resolution={"width": int(normalized_width), "height": int(normalized_height)},
        original_intrinsics=_intrinsics_dict(intrinsics),
        normalized_intrinsics=normalized_intrinsics,
        transform_meta=transform_meta,
    )
