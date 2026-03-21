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


def _bool_text(value):
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def build_summary_lines(traj_metrics, image_metrics, slam_metrics):
    traj = traj_metrics or {}
    image = image_metrics or {}
    slam = slam_metrics or {}

    lines = [
        "=== Trajectory Eval ===",
        "status: {}".format(traj.get("eval_status", "failed")),
        "APE RMSE: {}".format(fmt_value(traj.get("ape_trans", {}).get("rmse"), "m")),
        "RPE trans RMSE: {}".format(fmt_value(traj.get("rpe_trans", {}).get("rmse"), "m")),
        "RPE rot RMSE: {}".format(fmt_value(traj.get("rpe_rot", {}).get("rmse"), "deg")),
        "matched poses: {}".format(int(traj.get("matched_pose_count") or 0)),
        "ori_path_length_m: {}".format(fmt_value(traj.get("ori_path_length_m"), "m")),
        "gen_path_length_m: {}".format(fmt_value(traj.get("gen_path_length_m"), "m")),
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
    lines.extend(
        [
            "",
            "=== SLAM-Friendly Image Eval ===",
            "status: {}".format(slam.get("eval_status", "failed")),
            "reference_source: {}".format(slam.get("reference_source", "unavailable")),
            "uses_proxy_reference: {}".format(_bool_text(slam.get("uses_proxy_reference"))),
            "inlier_ratio mean / median / min / max: {} / {} / {} / {}".format(
                fmt_value(slam.get("inlier_ratio", {}).get("mean")),
                fmt_value(slam.get("inlier_ratio", {}).get("median")),
                fmt_value(slam.get("inlier_ratio", {}).get("min")),
                fmt_value(slam.get("inlier_ratio", {}).get("max")),
            ),
            "pose_success_rate: {}".format(fmt_value(slam.get("pose_success_rate"))),
            "valid_pair_count: {}".format(int(slam.get("valid_pair_count") or 0)),
            "valid_pose_pair_count: {}".format(int(slam.get("valid_pose_pair_count") or 0)),
            "successful_pose_pair_count: {}".format(int(slam.get("successful_pose_pair_count") or 0)),
        ]
    )
    lines.extend(_warning_lines(slam))
    return lines


def write_eval_summary(out_dir, traj_metrics, image_metrics, slam_metrics):
    out_path = Path(out_dir).resolve() / "summary.txt"
    write_text(out_path, "\n".join(build_summary_lines(traj_metrics, image_metrics, slam_metrics)) + "\n")
    return out_path


def log_eval_terminal_summary(traj_metrics, image_metrics, slam_metrics, out_dir):
    traj_status = str((traj_metrics or {}).get("eval_status", "failed"))
    matched = int((traj_metrics or {}).get("matched_pose_count") or 0)
    prefix = log_info if traj_status in ("success", "partial") else log_warn

    prefix("eval summary: traj/image/slam evaluation completed")
    log_info("eval traj: status={} matched_poses={}".format(traj_status, matched))
    log_info("eval traj: APE RMSE={}".format(fmt_value((traj_metrics or {}).get("ape_trans", {}).get("rmse"), "m")))
    log_info("eval traj: RPE trans RMSE={}".format(fmt_value((traj_metrics or {}).get("rpe_trans", {}).get("rmse"), "m")))

    image_status = str((image_metrics or {}).get("eval_status", "failed"))
    log_info(
        "eval image: status={} frame_count={}".format(
            image_status,
            int((image_metrics or {}).get("frame_count") or 0),
        )
    )
    log_info("eval image: PSNR mean={}".format(fmt_value((image_metrics or {}).get("psnr", {}).get("mean"), "dB")))
    log_info("eval image: MS-SSIM mean={}".format(fmt_value((image_metrics or {}).get("ms_ssim", {}).get("mean"))))
    log_info("eval image: LPIPS mean={}".format(_lpips_display(image_metrics)))

    slam_status = str((slam_metrics or {}).get("eval_status", "failed"))
    log_info(
        "eval slam: status={} valid_pairs={} ref={}".format(
            slam_status,
            int((slam_metrics or {}).get("valid_pair_count") or 0),
            str((slam_metrics or {}).get("reference_source", "unavailable")),
        )
    )
    log_info("eval slam: inlier_ratio mean={}".format(fmt_value((slam_metrics or {}).get("inlier_ratio", {}).get("mean"))))
    log_info("eval slam: pose_success_rate={}".format(fmt_value((slam_metrics or {}).get("pose_success_rate"))))
