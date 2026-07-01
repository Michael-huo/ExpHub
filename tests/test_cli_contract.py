from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from exphub import cli
from exphub.execution_plan import build_execution_plan


BASE_TRAIN = ["--mode", "train", "--dataset", "dummy", "--tag", "dummy", "--fps", "1"]
BASE_INFER = [
    "--mode",
    "infer",
    "--dataset",
    "dummy",
    "--sequence",
    "seq",
    "--tag",
    "dummy",
    "--fps",
    "1",
    "--start",
    "0",
    "--dur",
    "1",
]


class CliContractTests(unittest.TestCase):
    def parse_cli(self, argv):
        return cli._parse_args_and_plan(argv)

    def test_default_step_and_seed(self):
        run_config, plan = self.parse_cli(BASE_INFER)
        self.assertIsNone(plan.requested_step)
        self.assertEqual(plan.resolved_step, "all")
        self.assertEqual(run_config.seed, 12345)

    def test_decode_profile_and_log_level_are_kebab_case(self):
        run_config, _plan = self.parse_cli(
            BASE_INFER + ["--decode-profile", "lora_gdut", "--log-level", "quiet"]
        )
        self.assertEqual(run_config.decode_profile, "lora_gdut")
        self.assertEqual(run_config.log_level, "quiet")

    def test_train_input_contract_preserved(self):
        run_config, plan = self.parse_cli(BASE_TRAIN)
        self.assertEqual(run_config.sequence, "")
        self.assertEqual(run_config.start, "")
        self.assertEqual(run_config.dur, "")
        self.assertEqual(plan.stages, ("prepare", "encode", "lora"))

        for option in ("--start", "--dur"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit):
                    self.parse_cli(BASE_TRAIN + [option, "1"])

    def test_run_config_contains_only_retained_cli_values(self):
        run_config, _plan = self.parse_cli(BASE_INFER)
        self.assertEqual(
            set(run_config.__dataclass_fields__),
            {"dataset", "sequence", "tag", "fps", "start", "dur", "seed", "decode_profile", "log_level"},
        )
        for legacy_field in (
            "keep_level",
            "lora_resume",
            "segment_policy",
            "gpus",
            "exp_root",
            "droid_repo",
            "droid_weights",
            "encode_motion_benchmark",
            "compression_benchmark",
            "decode_image_quality",
        ):
            with self.subTest(legacy_field=legacy_field):
                self.assertFalse(hasattr(run_config, legacy_field))

    def test_experiments_parse_and_record_only_for_infer_all(self):
        _run_config, plan = self.parse_cli(BASE_INFER + ["--experiments", "motion-benchmark", "image-quality"])
        self.assertEqual(plan.experiments, ("motion-benchmark", "image-quality"))

    def test_experiments_without_values_is_parser_error(self):
        with self.assertRaises(SystemExit):
            self.parse_cli(BASE_INFER + ["--experiments"])

    def test_duplicate_experiment_is_plan_error(self):
        with self.assertRaises(SystemExit):
            self.parse_cli(BASE_INFER + ["--experiments", "motion-benchmark", "motion-benchmark"])

    def test_negative_seed_is_plan_error(self):
        with self.assertRaises(SystemExit):
            self.parse_cli(BASE_INFER + ["--seed", "-1"])

    def test_invalid_plan_does_not_build_runtime_or_load_platform(self):
        with mock.patch("exphub.cli.build_runtime") as build_runtime:
            with mock.patch("exphub.config.get_platform_config") as get_platform_config:
                with self.assertRaises(SystemExit):
                    cli.main(BASE_TRAIN + ["--step", "decode"])
        build_runtime.assert_not_called()
        get_platform_config.assert_not_called()

    def test_non_eval_completion_does_not_read_stale_eval_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "run"
            (exp_dir / "eval").mkdir(parents=True)
            (exp_dir / "eval" / "summary.json").write_text('{"payload": {"reduction_pct": 99}}', encoding="utf-8")
            (exp_dir / "eval" / "summary.csv").write_text("reduction_pct\n99\n", encoding="utf-8")

            runtime = mock.Mock()
            runtime.spec.dataset = "dummy"
            runtime.spec.sequence = "seq"
            runtime.spec.tag = "dummy"
            runtime.fps_arg = "1"
            runtime.paths.exp_dir = exp_dir
            result = mock.Mock()
            result.exp_dir = exp_dir
            result.step_times = {"decode": 0.1}

            with mock.patch("exphub.cli.build_runtime", return_value=runtime):
                with mock.patch("exphub.cli.run_runtime", return_value=result):
                    with mock.patch("exphub.cli._read_json_dict", side_effect=AssertionError("stale eval read")):
                        output = io.StringIO()
                        with redirect_stdout(output):
                            cli.main(BASE_INFER + ["--step", "decode"])

            text = output.getvalue()
            self.assertIn("RUN COMPLETE", text)
            self.assertIn("Stages", text)
            self.assertIn("decode", text)
            self.assertIn(str(exp_dir), text)
            self.assertNotIn("99", text)

    def test_eval_completion_prints_canonical_metrics_and_optional_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "run"
            (exp_dir / "eval" / "compression_benchmark").mkdir(parents=True)
            (exp_dir / "encode").mkdir(parents=True)
            (exp_dir / "decode").mkdir(parents=True)
            (exp_dir / "eval" / "summary.json").write_text(
                json.dumps(
                    {
                        "payload": {
                            "raw_bytes": 10485760,
                            "payload_bytes": 1048576,
                            "ratio": 0.1,
                            "reduction_pct": 90,
                            "raw_frame_count": 100,
                            "transmitted_frame_count": 10,
                            "unit_count": 4,
                        },
                        "vslam": {
                            "ori_ape_rmse_m": 0.1,
                            "rec_ape_rmse_m": 0.12,
                            "ape_delta_rec_minus_ori_m": 0.02,
                        },
                        "runtime": {
                            "prepare_time_s": 1,
                            "encode_time_s": 2,
                            "decode_time_s": 3,
                            "eval_time_s": 4,
                            "main_pipeline_time_s": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "encode" / "motion_benchmark").mkdir(parents=True)
            (exp_dir / "decode" / "image_quality").mkdir(parents=True)
            (exp_dir / "encode" / "motion_benchmark" / "summary.json").write_text(
                json.dumps(
                    {
                        "methods": {
                            "phase_correlation": {
                                "valid_rate": 1.0,
                                "total_time_s": 0.2,
                                "avg_time_ms_per_pair": 2.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "eval" / "compression_benchmark" / "summary.json").write_text(
                json.dumps(
                    {
                        "methods": {
                            "raw": {
                                "status": "ok",
                                "payload_bytes": 10485760,
                                "reduction_pct": 0,
                                "ape_rmse_m": 0.1,
                            },
                            "h265": {
                                "status": "failed",
                                "payload_bytes": 1048576,
                                "reduction_pct": 90,
                                "ape_rmse_m": None,
                                "error_message": "No module named 'torch'",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "decode" / "image_quality" / "summary.json").write_text(
                json.dumps(
                    {
                        "row": {
                            "matched_frame_count": 10,
                            "evaluated_frame_count": 10,
                            "lpips": 0.1,
                            "ssim": 0.9,
                            "fid": 3.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = mock.Mock()
            result.exp_dir = exp_dir
            result.step_times = {}
            result.main_pipeline_wall_time_s = None
            result.selected_stages_wall_time_s = None
            result.experiment_times = {}
            result.optional_total_time_s = None
            result.full_command_wall_time_s = None
            plan = build_execution_plan(
                mode="infer",
                requested_step=None,
                experiments=("motion-benchmark", "compression-benchmark", "image-quality"),
                seed=12345,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                cli._print_completion_report(result, plan)

            text = output.getvalue()
            self.assertNotIn("[Stage Times]", text)
            self.assertIn("[Payload]", text)
            self.assertIn("10.00 MiB", text)
            self.assertIn("APE REC-ORI", text)
            self.assertIn("+0.0200 m", text)
            self.assertNotIn("R" + "PE", text)
            self.assertIn("[Motion Benchmark]", text)
            self.assertIn("[Compression Benchmark]", text)
            self.assertIn("APE=0.1000 m", text)
            self.assertIn("status=failed", text)
            self.assertIn("No module named 'torch'", text)
            self.assertIn("[Image Quality]", text)

    def test_optional_experiment_times_follow_execution_plan_order(self):
        result = mock.Mock()
        result.step_times = {}
        result.main_pipeline_wall_time_s = None
        result.selected_stages_wall_time_s = None
        result.full_command_wall_time_s = None
        result.optional_total_time_s = 6.0
        result.experiment_times = {
            "compression-benchmark": 2.0,
            "image-quality": 3.0,
            "motion-benchmark": 1.0,
        }
        plan = build_execution_plan(
            mode="infer",
            requested_step=None,
            experiments=("motion-benchmark", "compression-benchmark", "image-quality"),
            seed=12345,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            cli._print_runtime_times(result, plan)

        text = output.getvalue()
        motion_pos = text.index("motion benchmark time")
        compression_pos = text.index("compression benchmark time")
        image_pos = text.index("image quality time")
        total_pos = text.index("optional total time")
        self.assertLess(motion_pos, compression_pos)
        self.assertLess(compression_pos, image_pos)
        self.assertLess(image_pos, total_pos)

    def test_main_keeps_step_progress_and_final_report_without_intermediate_experiment_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "run"
            (exp_dir / "eval" / "compression_benchmark").mkdir(parents=True)
            (exp_dir / "encode").mkdir(parents=True)
            (exp_dir / "decode").mkdir(parents=True)

            runtime = mock.Mock()
            runtime.spec.dataset = "dummy"
            runtime.spec.sequence = "seq"
            runtime.spec.tag = "dummy"
            runtime.fps_arg = "1"
            runtime.paths.exp_dir = exp_dir

            result = mock.Mock()
            result.exp_dir = exp_dir
            result.step_times = {}

            def write_final_summaries(*_args, **_kwargs):
                print("[STEP] prepare start mode=infer step=all")
                print("[STEP] eval done sec=1.00 out=eval/")
                (exp_dir / "eval" / "summary.json").write_text(
                    json.dumps(
                        {
                            "payload": {"raw_bytes": 100, "payload_bytes": 10, "ratio": 0.1, "reduction_pct": 90},
                            "vslam": {"ori_ape_rmse_m": 0.1, "rec_ape_rmse_m": 0.2},
                            "runtime": {"eval_time_s": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )
                (exp_dir / "encode" / "motion_benchmark").mkdir(parents=True)
                (exp_dir / "decode" / "image_quality").mkdir(parents=True)
                (exp_dir / "encode" / "motion_benchmark" / "summary.json").write_text(
                    json.dumps({"methods": {"phase_correlation": {"valid_rate": 1.0}}}),
                    encoding="utf-8",
                )
                (exp_dir / "eval" / "compression_benchmark" / "summary.json").write_text(
                    json.dumps({"methods": {"raw": {"payload_bytes": 100, "reduction_pct": 0}}}),
                    encoding="utf-8",
                )
                (exp_dir / "decode" / "image_quality" / "summary.json").write_text(
                    json.dumps({"row": {"matched_frame_count": 1, "evaluated_frame_count": 1}}),
                    encoding="utf-8",
                )
                return result

            output = io.StringIO()
            with mock.patch("exphub.cli.build_runtime", return_value=runtime):
                with mock.patch("exphub.cli.run_runtime", side_effect=write_final_summaries):
                    with redirect_stdout(output):
                        cli.main(BASE_INFER + ["--experiments", "motion-benchmark", "compression-benchmark", "image-quality"])

            text = output.getvalue()
            self.assertIn("[STEP] prepare start", text)
            self.assertIn("RUN COMPLETE", text)
            self.assertIn("[Motion Benchmark]", text)
            self.assertIn("[Compression Benchmark]", text)
            self.assertIn("[Image Quality]", text)
            self.assertNotIn("EXPERIMENT " + "SUMMARY", text)
            self.assertNotIn("phase_correlation." + "runtime_sec", text)
            self.assertNotIn("[Compression Benchmark: encode] " + "Method " + "Summary", text)
            self.assertNotIn("[Compression Benchmark: decode] " + "Method " + "Summary", text)
            self.assertNotIn("[Compression Benchmark: eval] " + "Method " + "Summary", text)
            self.assertNotIn("matched_" + "frames", text)

    def test_legacy_and_removed_flags_are_rejected(self):
        removed_flags = [
            "--keep-level",
            "--lora-resume",
            "--infer-extra",
            "--decode-image-quality",
            "--decode-image-quality-stride",
            "--decode-image-quality-max-frames",
            "--decode-image-quality-device",
            "--encode-motion-benchmark",
            "--compression-benchmark",
            "--droid-seq",
            "--viz",
            "--no-viz",
            "--decode_profile",
            "--log_level",
        ]
        for flag in removed_flags:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    cli._build_arg_parser().parse_args(BASE_TRAIN + [flag])

    def test_long_option_abbreviations_are_rejected(self):
        for flag in ("--decode-prof", "--log-lev", "--exper"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    cli._build_arg_parser().parse_args(BASE_INFER + [flag, "x"])


if __name__ == "__main__":
    unittest.main()
