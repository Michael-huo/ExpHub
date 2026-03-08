#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from datetime import datetime

import numpy as np
from scripts._common import log_err, log_info, log_warn

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def load_calib(calib_in, fx, fy, cx, cy, dist):
    """Read intrinsics from calib.txt or scalar CLI args."""
    if calib_in:
        arr = np.loadtxt(calib_in, delimiter=" ")
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size < 4:
            raise ValueError("calib_in must have at least 4 numbers: fx fy cx cy")
        fx, fy, cx, cy = map(float, arr[:4])
        dist = arr[4:].astype(np.float64) if arr.size > 4 else None
        return fx, fy, cx, cy, dist

    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError("Need --calib_in OR provide --fx --fy --cx --cy")

    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    dist_arr = None
    if dist:
        dist_arr = np.array([float(x) for x in dist], dtype=np.float64)
    return fx, fy, cx, cy, dist_arr


def compute_center_crop_to_aspect(w, h, target_ratio):
    """Center-crop to target aspect ratio. Return (x0, y0, crop_w, crop_h)."""
    in_ratio = float(w) / float(h)
    if in_ratio > target_ratio:
        crop_h = h
        crop_w = int(round(h * target_ratio))
        x0 = int((w - crop_w) // 2)
        y0 = 0
    else:
        crop_w = w
        crop_h = int(round(w / target_ratio))
        x0 = 0
        y0 = int((h - crop_h) // 2)

    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))
    return x0, y0, crop_w, crop_h


def build_spatial_transform(input_w, input_h, target_w, target_h):
    """Build the legacy crop-resize-crop transform."""
    target_ratio = float(target_w) / float(target_h)
    x0, y0, cw, ch = compute_center_crop_to_aspect(input_w, input_h, target_ratio)

    sx = float(target_w) / float(cw)
    sy = float(target_h) / float(ch)
    s = max(sx, sy)

    rw = int(round(cw * s))
    rh = int(round(ch * s))
    rw = max(rw, target_w)
    rh = max(rh, target_h)

    x1 = int((rw - target_w) // 2)
    y1 = int((rh - target_h) // 2)

    return {
        "input_w": int(input_w),
        "input_h": int(input_h),
        "target_w": int(target_w),
        "target_h": int(target_h),
        "crop1_x0": int(x0),
        "crop1_y0": int(y0),
        "crop1_w": int(cw),
        "crop1_h": int(ch),
        "scale": float(s),
        "resized_w": int(rw),
        "resized_h": int(rh),
        "crop2_x0": int(x1),
        "crop2_y0": int(y1),
        "scale_x_ideal": float(sx),
        "scale_y_ideal": float(sy),
        "note": "pipeline: crop1(center->aspect) -> resize(isotropic scale=max(sx,sy)) -> crop2(center->exact target)",
    }


def apply_spatial_transform(img_bgr, tfm, interpolation):
    x0, y0, cw, ch = tfm["crop1_x0"], tfm["crop1_y0"], tfm["crop1_w"], tfm["crop1_h"]
    s = tfm["scale"]
    rw, rh = tfm["resized_w"], tfm["resized_h"]
    x1, y1 = tfm["crop2_x0"], tfm["crop2_y0"]
    tw, th = tfm["target_w"], tfm["target_h"]

    crop1 = img_bgr[y0:y0 + ch, x0:x0 + cw]

    if interpolation == "auto":
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
    elif interpolation == "area":
        interp = cv2.INTER_AREA
    elif interpolation == "lanczos":
        interp = cv2.INTER_LANCZOS4
    elif interpolation == "linear":
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_LANCZOS4

    resized = cv2.resize(crop1, (rw, rh), interpolation=interp)
    crop2 = resized[y1:y1 + th, x1:x1 + tw]
    return crop2


def transform_intrinsics(fx, fy, cx, cy, tfm):
    x0, y0 = tfm["crop1_x0"], tfm["crop1_y0"]
    s = tfm["scale"]
    x1, y1 = tfm["crop2_x0"], tfm["crop2_y0"]

    cx1 = (cx - x0) * s
    cy1 = (cy - y0) * s
    fx1 = fx * s
    fy1 = fy * s

    cx2 = cx1 - x1
    cy2 = cy1 - y1
    return float(fx1), float(fy1), float(cx2), float(cy2)


def _decode_compressed_image(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _decode_raw_image(msg):
    """Decode sensor_msgs/Image to BGR uint8 without cv_bridge."""
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step) if hasattr(msg, "step") else w
    enc = (msg.encoding or "").lower()

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < step * h:
        raise ValueError("raw image buffer too small: {} < {}*{}".format(buf.size, step, h))
    buf = buf[: step * h].reshape((h, step))

    def take(nbytes_per_pixel, nch):
        need = w * nbytes_per_pixel
        if step < need:
            raise ValueError("step too small for encoding {}: step={} need={}".format(enc, step, need))
        arr = buf[:, :need].reshape((h, w, nch))
        return arr

    if enc in ("bgr8",):
        img = take(3, 3)
        return img.copy()
    if enc in ("rgb8",):
        img = take(3, 3)
        return img[:, :, ::-1].copy()
    if enc in ("mono8", "8uc1"):
        img = take(1, 1).reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8",):
        img = take(4, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        img = take(4, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    raise ValueError("unsupported sensor_msgs/Image encoding: {}".format(msg.encoding))


def decode_ros_image_to_bgr(msg):
    """Decode ROS image message (CompressedImage or Image) to BGR uint8."""
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "encoding"):
        return _decode_compressed_image(msg)

    if hasattr(msg, "encoding") and hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
        return _decode_raw_image(msg)

    raise TypeError("unknown ROS image message type: {}".format(getattr(msg, "_type", type(msg))))


def maybe_progress(iterable, quiet):
    if quiet or tqdm is None:
        return iterable
    return tqdm(iterable, desc="Extract", bar_format=_BAR_FORMAT)


def prepare_extraction(args, out_dir, frames_dir, root_dir):
    fx, fy, cx, cy, dist = load_calib(args.calib_in, args.fx, args.fy, args.cx, args.cy, args.dist)

    try:
        import rosbag
        from genpy import Time
    except Exception as e:
        log_err("import rosbag/genpy failed. Run in ROS1 env (source /opt/ros/<distro>/setup.bash).")
        log_err("error: {}".format(e))
        sys.exit(2)

    if not os.path.exists(args.bag):
        log_err("bag not found: {}".format(args.bag))
        sys.exit(2)

    bag = rosbag.Bag(args.bag, "r")

    first_time = None
    first_msg = None
    for _topic, _msg, _t in bag.read_messages(topics=[args.topic]):
        first_time = float(_t.to_sec())
        first_msg = _msg
        break
    if first_time is None:
        log_err("No messages found for topic: {}".format(args.topic))
        sys.exit(2)

    if (not args.quiet) and first_msg is not None:
        mt = getattr(first_msg, "_type", first_msg.__class__.__name__)
        enc = getattr(first_msg, "encoding", getattr(first_msg, "format", ""))
        w0 = getattr(first_msg, "width", None)
        h0 = getattr(first_msg, "height", None)
        if w0 is not None and h0 is not None:
            log_info("first_msg: type={} encoding/format={} size={}x{}".format(mt, enc, w0, h0))
        else:
            log_info("first_msg: type={} encoding/format={}".format(mt, enc))

    if args.start_abs is not None:
        start_time_abs = float(args.start_abs)
    elif args.start_idx >= 0:
        start_time_abs = None
        idx = 0
        for _topic, _msg, _t in bag.read_messages(topics=[args.topic]):
            if idx == args.start_idx:
                start_time_abs = float(_t.to_sec())
                break
            idx += 1
        if start_time_abs is None:
            log_err("start_idx {} out of range for topic {}".format(args.start_idx, args.topic))
            sys.exit(2)
    else:
        start_time_abs = first_time + float(args.start_sec)

    duration = float(args.duration)
    fps = float(args.fps)
    if fps <= 0 or duration <= 0:
        raise ValueError("fps/duration must be > 0")

    out_count = int(math.floor(duration * fps + 1e-9)) + 1
    dt = 1.0 / fps
    target_abs_times = [start_time_abs + i * dt for i in range(out_count)]
    target_rel_times = [i * dt for i in range(out_count)]
    end_time_abs = start_time_abs + (out_count - 1) * dt

    if not args.quiet:
        log_info("out_dir: {}".format(out_dir))
        log_info("start_time_abs: {:.6f}  end_time_abs: {:.6f}".format(start_time_abs, end_time_abs))
        log_info("out_count: {}  fps: {:.3f}".format(out_count, fps))

    margin = 2.0
    st = Time.from_sec(start_time_abs - margin)
    et = Time.from_sec(end_time_abs + margin)

    return {
        "out_dir": out_dir,
        "frames_dir": frames_dir,
        "root_dir": root_dir,
        "bag": bag,
        "first_time": float(first_time),
        "first_msg": first_msg,
        "window_start": st,
        "window_end": et,
        "start_time_abs": float(start_time_abs),
        "end_time_abs": float(end_time_abs),
        "duration": float(duration),
        "fps": float(fps),
        "out_count": int(out_count),
        "target_abs_times": target_abs_times,
        "target_rel_times": target_rel_times,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "dist": dist,
        "spatial_transform": None,
        "preprocess_meta": None,
        "calib_out": None,
    }


def iter_processed_frames(args, ctx):
    bag = ctx["bag"]
    prev_t = None
    prev_msg = None
    k = 0
    last_src_stamp = None
    last_src_img = None

    try:
        iterator = bag.read_messages(
            topics=[args.topic],
            start_time=ctx["window_start"],
            end_time=ctx["window_end"],
        )
        for _topic, msg, t in maybe_progress(iterator, args.quiet):
            cur_t = float(t.to_sec())

            if prev_t is None:
                prev_t = cur_t
                prev_msg = msg
                continue

            while k < ctx["out_count"] and ctx["target_abs_times"][k] <= cur_t:
                t_k = ctx["target_abs_times"][k]

                if prev_msg is None:
                    chosen_t, chosen_msg = cur_t, msg
                else:
                    if abs(t_k - prev_t) <= abs(cur_t - t_k):
                        chosen_t, chosen_msg = prev_t, prev_msg
                    else:
                        chosen_t, chosen_msg = cur_t, msg

                if last_src_stamp is not None and abs(chosen_t - last_src_stamp) < 1e-12:
                    img_bgr = last_src_img
                else:
                    img_bgr = decode_ros_image_to_bgr(chosen_msg)
                    last_src_stamp = chosen_t
                    last_src_img = img_bgr

                if img_bgr is None:
                    log_warn("failed to decode image at t={:.6f}".format(chosen_t))
                    k += 1
                    continue

                if ctx["spatial_transform"] is None:
                    ih, iw = img_bgr.shape[:2]
                    tfm = build_spatial_transform(iw, ih, args.width, args.height)
                    fx2, fy2, cx2, cy2 = transform_intrinsics(ctx["fx"], ctx["fy"], ctx["cx"], ctx["cy"], tfm)

                    calib_out = [fx2, fy2, cx2, cy2]
                    dist = ctx["dist"]
                    if dist is not None and dist.size > 0:
                        calib_out.extend([float(x) for x in dist])

                    meta = {
                        "version": args.version,
                        "method": "cropResize",
                        "bag": os.path.abspath(args.bag),
                        "topic": args.topic,
                        "strategy": args.strategy,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "first_time_abs": float(ctx["first_time"]),
                        "start_time_abs": float(ctx["start_time_abs"]),
                        "duration_sec": float(ctx["duration"]),
                        "fps": float(ctx["fps"]),
                        "output_count": int(ctx["out_count"]),
                        "spatial": tfm,
                        "intrinsics_in": {
                            "fx": ctx["fx"],
                            "fy": ctx["fy"],
                            "cx": ctx["cx"],
                            "cy": ctx["cy"],
                            "dist": dist.tolist() if dist is not None else [],
                        },
                        "intrinsics_out": {
                            "fx": fx2,
                            "fy": fy2,
                            "cx": cx2,
                            "cy": cy2,
                            "dist": dist.tolist() if dist is not None else [],
                        },
                    }
                    ctx["spatial_transform"] = tfm
                    ctx["preprocess_meta"] = meta
                    ctx["calib_out"] = calib_out

                out_img = apply_spatial_transform(img_bgr, ctx["spatial_transform"], args.interpolation)
                yield {
                    "index": int(k),
                    "image": out_img,
                    "timestamp_line": "{:.9f}\n".format(ctx["target_rel_times"][k]),
                }
                k += 1

            prev_t = cur_t
            prev_msg = msg
            if k >= ctx["out_count"]:
                break

        if k < ctx["out_count"] and prev_msg is not None:
            if not args.quiet:
                log_warn("padding remaining frames with last msg: {}/{}".format(k, ctx["out_count"]))
            img_bgr = decode_ros_image_to_bgr(prev_msg)
            while k < ctx["out_count"] and img_bgr is not None:
                out_img = apply_spatial_transform(img_bgr, ctx["spatial_transform"], args.interpolation)
                yield {
                    "index": int(k),
                    "image": out_img,
                    "timestamp_line": "{:.9f}\n".format(ctx["target_rel_times"][k]),
                }
                k += 1
    finally:
        bag.close()
