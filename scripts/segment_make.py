#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_make.py

从 ROS1 rosbag（sensor_msgs/Image 或 sensor_msgs/CompressedImage）制作“标准化片段（segment）”：
- 统一输出目录结构：
  <out_dir>/
    frames/000000.png ...
    keyframes/000000.png ...   # (optional) anchors every kf_gap frames (symlink/hardlink/copy)
    timestamps.txt        # 相对时间(s)，每行一个
    calib.txt             # 对应 frames 分辨率的 K（fx fy cx cy [dist...]）
    preprocess_meta.json  # 裁剪/缩放参数（含 2 段裁剪 + 等比缩放）
    keyframes/keyframes_meta.json  # (optional) keyframe sampling meta

空间规范（你已确认采用）：
- 先中心裁剪到目标宽高比
- 再“等比缩放”（用单一 scale s）
- 由于整数像素四舍五入导致的微小误差，可能会在缩放后再做一次中心裁剪以精确到目标 W×H
  （这一步一般会是 0，但我们把它显式记录，保证 K 与图像严格一致）

时间规范：
- 使用 rosbag 的消息时间戳，重采样到目标 fps 的均匀时间网格（默认 nearest）
- timestamps.txt 存相对时间（从 0.0 开始）

依赖：
- ROS1 环境（能 import rosbag, genpy）
- opencv-python, numpy
"""

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from _common import list_frames_sorted, log_err, log_info, log_prog, log_warn, write_json_atomic
from _segment.api import materialize_keyframe_plan

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)

# tqdm 可选：没有也能跑
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _dir_file_stats(dir_path):
    """Return (file_count, bytes_sum) for direct files under a directory."""
    if not os.path.isdir(dir_path):
        return 0, 0

    file_count = 0
    bytes_sum = 0
    for name in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, name)
        if not os.path.isfile(fp):
            continue
        file_count += 1
        try:
            bytes_sum += int(os.path.getsize(fp))
        except Exception:
            pass
    return int(file_count), int(bytes_sum)




def load_calib(calib_in, fx, fy, cx, cy, dist):
    """读取内参。约定 calib.txt 一行空格分隔：fx fy cx cy [dist...]"""
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
    """中心裁剪到目标宽高比。返回 (x0,y0,crop_w,crop_h)"""
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
    """
    构建稳定的“等比缩放”空间变换：
      crop1（中心裁剪到目标宽高比） -> scale（单一 s） -> crop2（中心裁剪到精确 target 尺寸）
    """
    target_ratio = float(target_w) / float(target_h)
    x0, y0, cw, ch = compute_center_crop_to_aspect(input_w, input_h, target_ratio)

    sx = float(target_w) / float(cw)
    sy = float(target_h) / float(ch)

    # 用 max 保证缩放后尺寸 >= target，便于 crop2 精确裁到目标尺寸
    s = max(sx, sy)

    rw = int(round(cw * s))
    rh = int(round(ch * s))

    # 防止 rounding 导致比 target 小
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
        "note": "pipeline: crop1(center->aspect) -> resize(isotropic scale=max(sx,sy)) -> crop2(center->exact target)"
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
    # CompressedImage: msg.data holds encoded bytes (jpeg/png)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _decode_raw_image(msg):
    """Decode sensor_msgs/Image to BGR uint8 without cv_bridge."""
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step) if hasattr(msg, "step") else w
    enc = (msg.encoding or "").lower()

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # Some drivers pad each row to 'step' bytes.
    if buf.size < step * h:
        raise ValueError(f"raw image buffer too small: {buf.size} < {step}*{h}")
    buf = buf[: step * h].reshape((h, step))

    def take(nbytes_per_pixel, nch):
        need = w * nbytes_per_pixel
        if step < need:
            raise ValueError(f"step too small for encoding {enc}: step={step} need={need}")
        arr = buf[:, :need].reshape((h, w, nch))
        return arr

    if enc in ("bgr8",):
        img = take(3, 3)
        return img.copy()
    if enc in ("rgb8",):
        img = take(3, 3)
        return img[:, :, ::-1].copy()  # RGB -> BGR
    if enc in ("mono8", "8uc1"):
        img = take(1, 1).reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8",):
        img = take(4, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        img = take(4, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # 常见深度/其他格式：此版本不支持（实验阶段先聚焦 RGB 单目）
    raise ValueError(f"unsupported sensor_msgs/Image encoding: {msg.encoding}")


def decode_ros_image_to_bgr(msg):
    """Decode ROS image message (CompressedImage or Image) to BGR uint8."""
    # sensor_msgs/CompressedImage has fields: format, data
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "encoding"):
        return _decode_compressed_image(msg)

    # sensor_msgs/Image has fields: encoding, height, width, step, data
    if hasattr(msg, "encoding") and hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
        return _decode_raw_image(msg)

    raise TypeError(f"unknown ROS image message type: {getattr(msg, '_type', type(msg))}")


def maybe_progress(iterable, quiet):
    if quiet or tqdm is None:
        return iterable
    return tqdm(iterable, desc="Extract", bar_format=_BAR_FORMAT)


def main():
    ap = argparse.ArgumentParser(description="Make standardized dataset from rosbag (Image/CompressedImage).")

    ap.add_argument("--bag", required=True, help="rosbag path")
    ap.add_argument("--topic", default="/camera/rgb/image_raw/compressed", help="image topic (sensor_msgs/Image or sensor_msgs/CompressedImage)")
    ap.add_argument("--out_root", required=True, help="output root, e.g. <exphub>/datasets/<dataset>")

    ap.add_argument("--name", default="", help="segment folder name, e.g. scand_seq01_dur16s_w768_h480_fps25_v1")
    ap.add_argument("--dataset", default="scand", help="dataset name (used only when --name not set)")
    ap.add_argument("--seq", default="seq01", help="sequence label (used only when --name not set)")
    ap.add_argument("--version", default="v1", help="version suffix (used only when --name not set)")

    ap.add_argument("--duration", type=float, default=16.0, help="segment duration in seconds")
    ap.add_argument("--fps", type=float, default=25.0, help="target fps for resampling (Hz)")
    ap.add_argument("--start_idx", type=int, default=-1, help="start index in the image topic (0-based). If set, overrides --start_sec")
    ap.add_argument("--start_sec", type=float, default=0.0, help="start time offset (sec) from FIRST image msg time")
    ap.add_argument("--start_abs", type=float, default=None, help="absolute start time in seconds (ROS time). Overrides start_idx/start_sec")

    ap.add_argument("--width", type=int, required=True, help="target width")
    ap.add_argument("--height", type=int, required=True, help="target height")
    ap.add_argument("--interpolation", default="auto", choices=["auto", "area", "lanczos", "linear"], help="resize interpolation")

    ap.add_argument("--calib_in", default="", help="raw calib file (fx fy cx cy [dist...]) for original images")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--dist", nargs="*", default=None, help="optional distortion coeffs")

    ap.add_argument("--png_compress", type=int, default=1, help="0-9, smaller is faster/bigger file")
    ap.add_argument("--strategy", default="nearest", choices=["nearest"], help="resample strategy (v1: nearest)")

    # keyframes（为后续压缩率统计准备）：每隔 kf_gap 取一帧（包含 0）
    # 当 (N-1) 不能被 kf_gap 整除时，末尾会有少量 tail_drop 帧不参与后续 infer/merge。
    ap.add_argument("--kf_gap", type=int, default=0, help="keyframe gap in frames. if >0, will create keyframes/ folder under dataset root")
    ap.add_argument(
        "--keyframes_mode",
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="how to materialize keyframes in keyframes/: symlink (default, no duplication), hardlink, or copy",
    )
    ap.add_argument(
        "--segment_policy",
        default="uniform",
        choices=["uniform", "sks_v1", "semantic_guarded_v1", "semantic_guarded_v2"],
        help="keyframe policy: uniform legacy anchors, sks_v1 fixed-budget semantic relocation, or semantic_guarded_v1/v2 guarded refinements",
    )

    ap.add_argument("--dry_run", action="store_true", help="print plan and exit")
    ap.add_argument("--quiet", action="store_true", help="less logs")

    args = ap.parse_args()

    if args.width % 16 != 0 or args.height % 16 != 0:
        log_warn("target size {}x{} is not divisible by 16 (may affect diffusion models)".format(args.width, args.height))

    dur_tag = int(round(args.duration))
    fps_tag = int(round(args.fps))
    if args.name:
        ds_name = args.name
    else:
        ds_name = "{}_{}_dur{}s_w{}_h{}_fps{}_{}".format(
            args.dataset, args.seq, dur_tag, args.width, args.height, fps_tag, args.version
        )

    out_dir = os.path.join(args.out_root, ds_name)
    frames_dir = os.path.join(out_dir, "frames")
    # 不再创建 meta/ 子目录：所有非图像产物直接与 frames/ 同级放在数据集根目录
    root_dir = out_dir

    if args.dry_run:
        log_prog("dry run: out_dir={}".format(out_dir))
        log_info("dry run: topic={}".format(args.topic))
        # 关键：把终点帧包含进采样网格。
        # 例如 dur=4s,fps=24Hz => 0..4s 共 97 帧（0..96），才能形成 4 个完整 1s 区间锚点。
        out_cnt = int(math.floor(float(args.duration) * float(args.fps) + 1e-9)) + 1
        log_info("dry run: duration={} fps={} -> frames={}".format(args.duration, args.fps, out_cnt))
        return

    ensure_dir(frames_dir)
    ensure_dir(root_dir)

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

    # first message time
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


    # start time
    start_time_abs = None
    if args.start_abs is not None:
        start_time_abs = float(args.start_abs)
    elif args.start_idx >= 0:
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

    # 时间网格：包含终点（t=duration）。
    # out_count = floor(duration*fps) + 1
    # 这样当 duration*fps 为整数时，最后一帧严格落在 t=duration。
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

    k = 0
    prev_t = None
    prev_msg = None

    tfm = None
    last_src_stamp = None
    last_src_img = None
    ts_lines = []

    for _topic, msg, t in maybe_progress(bag.read_messages(topics=[args.topic], start_time=st, end_time=et), args.quiet):
        cur_t = float(t.to_sec())

        if prev_t is None:
            prev_t = cur_t
            prev_msg = msg
            continue

        while k < out_count and target_abs_times[k] <= cur_t:
            t_k = target_abs_times[k]

            if prev_msg is None:
                chosen_t, chosen_msg = cur_t, msg
            else:
                if abs(t_k - prev_t) <= abs(cur_t - t_k):
                    chosen_t, chosen_msg = prev_t, prev_msg
                else:
                    chosen_t, chosen_msg = cur_t, msg

            # decode with cache
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

            if tfm is None:
                ih, iw = img_bgr.shape[:2]
                tfm = build_spatial_transform(iw, ih, args.width, args.height)
                fx2, fy2, cx2, cy2 = transform_intrinsics(fx, fy, cx, cy, tfm)

                calib_out = [fx2, fy2, cx2, cy2]
                if dist is not None and dist.size > 0:
                    calib_out.extend([float(x) for x in dist])

                meta = {
                    "version": args.version,
                    "method": "cropResize",
                    "bag": os.path.abspath(args.bag),
                    "topic": args.topic,
                    "strategy": args.strategy,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "first_time_abs": float(first_time),
                    "start_time_abs": float(start_time_abs),
                    "duration_sec": float(duration),
                    "fps": float(fps),
                    "output_count": int(out_count),
                    "spatial": tfm,
                    "intrinsics_in": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist": dist.tolist() if dist is not None else []},
                    "intrinsics_out": {"fx": fx2, "fy": fy2, "cx": cx2, "cy": cy2, "dist": dist.tolist() if dist is not None else []},
                }
                with open(os.path.join(root_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                np.savetxt(os.path.join(root_dir, "calib.txt"),
                           np.array(calib_out, dtype=np.float64).reshape(1, -1), fmt="%.10f")

            out_img = apply_spatial_transform(img_bgr, tfm, args.interpolation)
            out_path = os.path.join(frames_dir, "{:06d}.png".format(k))
            cv2.imwrite(out_path, out_img, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compress)])

            t_rel = target_rel_times[k]
            delta = float(chosen_t - t_k)

            # 仅保留 timestamps.txt（后续 merge/droid 会用到）。
            # frame_map.csv 属于调试/溯源信息，本 clean 版本不再写出。
            ts_lines.append("{:.9f}\n".format(t_rel))
            k += 1

        prev_t = cur_t
        prev_msg = msg
        if k >= out_count:
            break

    # pad if needed
    if k < out_count and prev_msg is not None:
        if not args.quiet:
            log_warn("padding remaining frames with last msg: {}/{}".format(k, out_count))
        img_bgr = decode_ros_image_to_bgr(prev_msg)
        while k < out_count and img_bgr is not None:
            out_img = apply_spatial_transform(img_bgr, tfm, args.interpolation)
            out_path = os.path.join(frames_dir, "{:06d}.png".format(k))
            cv2.imwrite(out_path, out_img, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compress)])

            t_rel = target_rel_times[k]
            t_abs = target_abs_times[k]
            chosen_t = float(prev_t)
            delta = float(chosen_t - t_abs)

            ts_lines.append("{:.9f}\n".format(t_rel))
            k += 1

    bag.close()

    with open(os.path.join(root_dir, "timestamps.txt"), "w", encoding="utf-8") as f:
        f.writelines(ts_lines)

    # -------------------------
    # keyframes folder
    # -------------------------
    if int(args.kf_gap) > 0:
        kf_gap = int(args.kf_gap)
        log_info("segment policy materialize start: policy={} kf_gap={}".format(args.segment_policy, kf_gap))
        kf_meta = materialize_keyframe_plan(
            root_dir=root_dir,
            frames_dir=frames_dir,
            timestamps_path=os.path.join(root_dir, "timestamps.txt"),
            kf_gap=kf_gap,
            keyframes_mode=args.keyframes_mode,
            policy_name=args.segment_policy,
        )
        summary = dict(kf_meta.get("summary") or {})
        log_info(
            "segment policy materialize done: uniform_base={} final={} extra_ratio={:.3f}".format(
                int(summary.get("num_uniform_base", 0)),
                int(summary.get("num_final_keyframes", 0)),
                float(summary.get("extra_kf_ratio", 0.0)),
            )
        )

    # Step-level compact metadata (additional, non-breaking).
    actual_frame_count = len(list_frames_sorted(frames_dir))
    _, frames_bytes_sum = _dir_file_stats(frames_dir)
    keyframes_dir = os.path.join(root_dir, "keyframes")
    keyframes_file_count, keyframes_bytes_sum = _dir_file_stats(keyframes_dir)
    keyframes_frame_count = len(list_frames_sorted(keyframes_dir)) if os.path.isdir(keyframes_dir) else 0
    step_meta = {
        "step": "segment",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "bag": os.path.abspath(args.bag),
            "topic": args.topic,
        },
        "params": {
            "w": int(args.width),
            "h": int(args.height),
            "fps": float(args.fps),
            "dur": float(args.duration),
            "start_sec": float(args.start_sec),
            "start_idx": int(args.start_idx),
            "kf_gap": int(args.kf_gap),
            "segment_policy": str(args.segment_policy),
        },
        "outputs": {
            "frame_count": int(actual_frame_count),
            "bytes_sum": int(frames_bytes_sum),
            "timestamps_count": int(len(ts_lines)),
            "keyframes_frame_count": int(keyframes_frame_count),
            "keyframes_bytes_sum": int(keyframes_bytes_sum),
            "keyframes_file_count": int(keyframes_file_count),
            "keyframe_count": int(keyframes_frame_count),
            "keyframe_bytes_sum": int(keyframes_bytes_sum),
            "ori": {
                "frame_count": int(actual_frame_count),
                "bytes_sum": int(frames_bytes_sum),
            },
            "keyframes": {
                "frame_count": int(keyframes_frame_count),
                "bytes_sum": int(keyframes_bytes_sum),
            },
        },
    }
    if int(args.kf_gap) > 0 and 'kf_meta' in locals():
        step_meta["outputs"]["keyframe_policy"] = {
            "policy_name": str(kf_meta.get("policy_name", args.segment_policy)),
            "uniform_base_count": int((kf_meta.get("summary") or {}).get("num_uniform_base", 0)),
            "final_keyframe_count": int((kf_meta.get("summary") or {}).get("num_final_keyframes", 0)),
            "extra_kf_ratio": float((kf_meta.get("summary") or {}).get("extra_kf_ratio", 0.0)),
        }
    write_json_atomic(os.path.join(root_dir, "step_meta.json"), step_meta, indent=2)


    log_prog("wrote dataset: {}".format(out_dir))
    log_info("frames: {}".format(frames_dir))
    log_info("root: {}".format(root_dir))
    log_info("count: {}".format(out_count))


if __name__ == "__main__":
    main()
