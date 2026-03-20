#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from _common import log_info, log_warn
from _eval.io import fmt_value, write_text


def _warning_lines(metrics_obj):
    warnings_list = list((metrics_obj or {}).get("warnings", []) or [])
    if not warnings_list:
        return ["warnings: 0"]
    lines = ["warnings: {}".format(len(warnings_list))]
    for item in warnings_list:
        lines.append("- {}".format(item))
    return lines


def _lpips_display(image_metrics):
    value = (image_metrics or {}).get("lpips", {}).get("mean")
    if value is None:
        return "unavailable"
    return fmt_value(value)


def build_summary_lines(traj_metrics, image_metrics):
    traj = traj_metrics or {}
    image = image_metrics or {}

    lines = [
        "=== Trajectory Eval ===",
        "status: {}".format(traj.get("eval_status", "failed")),
        "APE RMSE: {}".format(fmt_value(traj.get("ape_trans", {}).get("rmse"), "m")),
        "RPE trans RMSE: {}".format(fmt_value(traj.get("rpe_trans", {}).get("rmse"), "m")),
        "RPE rot RMSE: {}".format(fmt_value(traj.get("rpe_rot", {}).get("rmse"), "deg")),
        "matched poses: {}".format(int(traj.get("matched_pose_count") or 0)),
    ]
    lines.extend(_warning_lines(traj))
    lines.extend(
        [
            "",
            "=== Image Eval ===",
            "status: {}".format(image.get("eval_status", "failed")),
            "PSNR mean / median / min / max: {} / {} / {} / {}".format(
                fmt_value(image.get("psnr", {}).get("mean"), "dB"),
                fmt_value(image.get("psnr", {}).get("median"), "dB"),
                fmt_value(image.get("psnr", {}).get("min"), "dB"),
                fmt_value(image.get("psnr", {}).get("max"), "dB"),
            ),
            "MS-SSIM mean / median / min / max: {} / {} / {} / {}".format(
                fmt_value(image.get("ms_ssim", {}).get("mean")),
                fmt_value(image.get("ms_ssim", {}).get("median")),
                fmt_value(image.get("ms_ssim", {}).get("min")),
                fmt_value(image.get("ms_ssim", {}).get("max")),
            ),
            "LPIPS mean / median / min / max: {} / {} / {} / {}".format(
                "n/a" if image.get("lpips", {}).get("mean") is None else fmt_value(image.get("lpips", {}).get("mean")),
                "n/a" if image.get("lpips", {}).get("median") is None else fmt_value(image.get("lpips", {}).get("median")),
                "n/a" if image.get("lpips", {}).get("min") is None else fmt_value(image.get("lpips", {}).get("min")),
                "n/a" if image.get("lpips", {}).get("max") is None else fmt_value(image.get("lpips", {}).get("max")),
            ),
            "frame_count: {}".format(int(image.get("frame_count") or 0)),
        ]
    )
    lines.extend(_warning_lines(image))
    return lines


def write_eval_summary(out_dir, traj_metrics, image_metrics):
    out_path = Path(out_dir).resolve() / "summary.txt"
    write_text(out_path, "\n".join(build_summary_lines(traj_metrics, image_metrics)) + "\n")
    return out_path


def log_eval_terminal_summary(traj_metrics, image_metrics, out_dir):
    traj_status = str((traj_metrics or {}).get("eval_status", "failed"))
    matched = int((traj_metrics or {}).get("matched_pose_count") or 0)
    prefix = log_info if traj_status in ("success", "partial") else log_warn

    prefix("eval traj: status={} matched_poses={} out={}".format(traj_status, matched, out_dir))
    prefix("eval traj: APE RMSE={}".format(fmt_value((traj_metrics or {}).get("ape_trans", {}).get("rmse"), "m")))
    prefix("eval traj: RPE trans RMSE={}".format(fmt_value((traj_metrics or {}).get("rpe_trans", {}).get("rmse"), "m")))

    image_status = str((image_metrics or {}).get("eval_status", "failed"))
    image_prefix = log_info if image_status in ("success", "partial") else log_warn
    image_prefix(
        "eval image: status={} frame_count={}".format(
            image_status,
            int((image_metrics or {}).get("frame_count") or 0),
        )
    )
    image_prefix("eval image: PSNR mean={}".format(fmt_value((image_metrics or {}).get("psnr", {}).get("mean"), "dB")))
    image_prefix("eval image: MS-SSIM mean={}".format(fmt_value((image_metrics or {}).get("ms_ssim", {}).get("mean"))))
    image_prefix("eval image: LPIPS mean={}".format(_lpips_display(image_metrics)))
