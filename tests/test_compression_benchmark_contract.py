from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from exphub.common.compression_benchmark import (
    benchmark_method_order,
    raw_payload_bytes_from_report,
    resolve_method_report,
)
from exphub.eval import compression_benchmark
from exphub.eval.compression_benchmark import run_compression_benchmark_eval
from output_capture import silent_stdio


PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_HEADER)


def _write_track(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0 0 0 1 0 0 0\n", encoding="utf-8")


def _fixture(root: Path):
    exp_dir = root / "run"
    prepare_frames = exp_dir / "prepare" / "raw_frames"
    _write_frame(prepare_frames / "000000.png")
    _write_track(exp_dir / "prepare" / "gt_traj.tum")
    _write_track(exp_dir / "eval" / "ori" / "traj_est.tum")
    _write_track(exp_dir / "eval" / "rec" / "traj_est.tum")
    for method in ("h265", "dcvc_fm_q21"):
        _write_frame(exp_dir / "decode" / "compression_benchmark" / method / "frames" / "000000.png")
    _write_json(
        exp_dir / "eval" / "summary.json",
        {
            "vslam": {
                "ori_ape_rmse_m": 0.111,
                "rec_ape_rmse_m": 0.222,
                "gt_path_length_m": 10.0,
            },
        },
    )
    _write_json(
        exp_dir / "decode" / "compression_benchmark" / "report.json",
        {
            "frame_count": 1,
            "fps": 24,
            "methods": {
                "raw": {"status": "ok", "payload_bytes": 1000, "frame_count": 1, "fps": 24, "reduction_pct": 0},
                "h265": {
                    "status": "ok",
                    "payload_bytes": 200,
                    "frame_count": 1,
                    "fps": 24,
                    "reduction_pct": 80,
                    "decoded_frames_dir": "decode/compression_benchmark/h265/frames",
                },
                "dcvc_fm_q21": {
                    "status": "ok",
                    "payload_bytes": 50,
                    "frame_count": 1,
                    "fps": 24,
                    "reduction_pct": 95,
                    "decoded_frames_dir": "decode/compression_benchmark/dcvc_fm_q21/frames",
                },
                "vlmem": {"status": "ok", "payload_bytes": 100, "frame_count": 1, "fps": 24, "reduction_pct": 90},
            },
        },
    )
    return exp_dir


def _config(exp_dir: Path):
    return {
        "exp_dir": str(exp_dir),
        "out_dir": str(exp_dir / "eval" / "compression_benchmark"),
        "decode_benchmark_report": str(
            exp_dir / "decode" / "compression_benchmark" / "report.json"
        ),
        "prepare_frames_dir": str(exp_dir / "prepare" / "raw_frames"),
        "main_eval_summary": str(exp_dir / "eval" / "summary.json"),
        "gt_traj": str(exp_dir / "prepare" / "gt_traj.tum"),
        "fps": 24,
        "t_max_diff": 0.03,
    }


class CompressionBenchmarkContractTests(unittest.TestCase):
    def test_canonical_raw_report_is_required(self):
        report = {
            "raw_frame_bytes": 1000,
            "methods_order": ["raw", "h265"],
            "methods": {
                "raw": {"method_key": "raw", "payload_bytes": 1000},
                "h265": {"method_key": "h265", "payload_bytes": 200},
            },
        }

        self.assertEqual(raw_payload_bytes_from_report(report), 1000)
        self.assertEqual(resolve_method_report(report, "raw")["payload_bytes"], 1000)
        self.assertEqual(benchmark_method_order(report)[:2], ["raw", "h265"])

    def test_legacy_zip_only_report_fails_clearly(self):
        legacy = {"methods": {"zip": {"method_key": "zip", "payload_bytes": 100}}}
        with self.assertRaisesRegex(RuntimeError, "canonical raw payload bytes"):
            raw_payload_bytes_from_report(legacy)
        with self.assertRaisesRegex(RuntimeError, "canonical raw method"):
            resolve_method_report(legacy, "raw")

    def test_summary_records_ape_for_mainline_and_single_track_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = _fixture(Path(tmp))
            ape_by_method = {"h265": 0.333, "dcvc_fm_q21": 0.444}

            def slam_side_effect(_config, method_key, _frames_dir):
                rel = "eval/compression_benchmark/{}/traj_est.tum".format(method_key)
                _write_track(exp_dir / rel)
                return {"trajectory_path": rel}

            def evo_side_effect(config):
                method_key = config["method_key"]
                summary_path = Path(config["out_dir"]) / "ape_summary.json"
                _write_json(summary_path, {"ape_rmse": ape_by_method[method_key], "gt_path_length_m": 10.0})
                return {
                    "summary_path": str(summary_path),
                    "summary": {"ape_rmse": ape_by_method[method_key], "gt_path_length_m": 10.0},
                }

            with mock.patch("exphub.eval.compression_benchmark.run_single_slam_track", side_effect=slam_side_effect) as slam:
                with mock.patch(
                    "exphub.eval.compression_benchmark.run_evo_eval_single_track",
                    side_effect=evo_side_effect,
                ) as evo:
                    with silent_stdio():
                        result = run_compression_benchmark_eval(_config(exp_dir))

            methods = result["summary"]["methods"]
            self.assertEqual(methods["raw"]["ape_rmse_m"], 0.111)
            self.assertEqual(methods["vlmem"]["ape_rmse_m"], 0.222)
            self.assertEqual(methods["h265"]["ape_rmse_m"], 0.333)
            self.assertEqual(methods["dcvc_fm_q21"]["ape_rmse_m"], 0.444)
            self.assertNotIn("r" + "pe" + "_trans_rmse_m", methods["h265"])
            self.assertEqual(slam.call_count, 2)
            self.assertEqual(evo.call_count, 2)

    def test_method_failure_keeps_status_and_error_without_fabricating_ape(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = _fixture(Path(tmp))

            def slam_side_effect(_config, method_key, _frames_dir):
                if method_key == "h265":
                    raise RuntimeError("No module named 'torch'")
                rel = "eval/compression_benchmark/{}/traj_est.tum".format(method_key)
                _write_track(exp_dir / rel)
                return {"trajectory_path": rel}

            def evo_side_effect(config):
                method_key = config["method_key"]
                summary_path = Path(config["out_dir"]) / "ape_summary.json"
                return {
                    "summary_path": str(summary_path),
                    "summary": {"ape_rmse": 0.444 if method_key == "dcvc_fm_q21" else 0.0, "gt_path_length_m": 10.0},
                }

            with mock.patch("exphub.eval.compression_benchmark.run_single_slam_track", side_effect=slam_side_effect):
                with mock.patch("exphub.eval.compression_benchmark.run_evo_eval_single_track", side_effect=evo_side_effect):
                    with silent_stdio():
                        result = run_compression_benchmark_eval(_config(exp_dir))

            h265 = result["summary"]["methods"]["h265"]
            self.assertEqual(h265["status"], "failed")
            self.assertIsNone(h265["ape_rmse_m"])
            self.assertIn("No module named 'torch'", h265["error_message"])

    def test_internal_entrypoint_passes_arguments_to_eval_runner(self):
        argv = [
            "--run-compression-benchmark-eval",
            "--exp_dir",
            "/tmp/run",
            "--out_dir",
            "/tmp/run/eval/compression_benchmark",
            "--decode_benchmark_report",
            "/tmp/run/decode/compression_benchmark/report.json",
            "--main_eval_summary",
            "/tmp/run/eval/summary.json",
            "--prepare_result",
            "/tmp/run/prepare/prepare_result.json",
            "--prepare_frames_dir",
            "/tmp/run/prepare/raw_frames",
            "--generation_units",
            "/tmp/run/encode/generation_units.json",
            "--encode_result",
            "/tmp/run/encode/encode_result.json",
            "--decode_frames_dir",
            "/tmp/run/decode/reconstructed_frames",
            "--decode_calib",
            "/tmp/run/decode/calib.txt",
            "--decode_timestamps",
            "/tmp/run/decode/timestamps.txt",
            "--decode_report",
            "/tmp/run/decode/decode_report.json",
            "--gt_traj",
            "/tmp/run/prepare/gt_traj.tum",
            "--droid_repo",
            "/opt/droid",
            "--weights",
            "/opt/droid/droid.pth",
            "--fps",
            "24",
            "--clip_duration",
            "20",
            "--t_max_diff",
            "0.03",
            "--disable_vis",
        ]
        with mock.patch("exphub.eval.compression_benchmark.run_compression_benchmark_eval", return_value={}) as runner:
            compression_benchmark.main(argv)

        config = runner.call_args.args[0]
        self.assertEqual(config["exp_dir"], "/tmp/run")
        self.assertEqual(config["out_dir"], "/tmp/run/eval/compression_benchmark")
        self.assertEqual(config["droid_repo"], "/opt/droid")
        self.assertEqual(config["weights"], "/opt/droid/droid.pth")
        self.assertEqual(config["fps"], 24.0)
        self.assertTrue(config["disable_vis"])


if __name__ == "__main__":
    unittest.main()
