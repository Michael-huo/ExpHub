from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from exphub.eval import eval as eval_pipeline
from exphub.execution_plan import build_execution_plan
from exphub import runner as runner_pipeline
from exphub.runner import RunConfig, build_runtime, run_runtime


class _EvalPaths:
    def __init__(self, root: Path):
        self.exp_dir = root
        self.prepare_dir = root / "prepare"
        self.prepare_result_path = self.prepare_dir / "prepare_result.json"
        self.prepare_frames_dir = self.prepare_dir / "raw_frames"
        self.prepare_gt_traj_path = self.prepare_dir / "gt_traj.tum"
        self.encode_generation_units_path = root / "encode" / "generation_units.json"
        self.encode_prompts_path = root / "encode" / "prompts.json"
        self.encode_result_path = root / "encode" / "encode_result.json"
        self.decode_frames_dir = root / "decode" / "reconstructed_frames"
        self.decode_calib_path = root / "decode" / "calib.txt"
        self.decode_timestamps_path = root / "decode" / "timestamps.txt"
        self.decode_report_path = root / "decode" / "decode_report.json"
        self.eval_dir = root / "eval"
        self.eval_canonical_summary_path = self.eval_dir / "summary.json"
        self.eval_canonical_summary_csv_path = self.eval_dir / "summary.csv"
        self.eval_trajectory_overlay_path = self.eval_dir / "trajectory_overlay_auto2d.png"
        self.eval_trajectory_interactive_path = self.eval_dir / "trajectory_overlay_interactive.html"
        self.eval_ori_traj_path = self.eval_dir / "ori" / "traj_est.tum"
        self.eval_rec_traj_path = self.eval_dir / "rec" / "traj_est.tum"
        self.eval_ori_run_meta_path = self.eval_dir / "ori" / "run_meta.json"
        self.eval_rec_run_meta_path = self.eval_dir / "rec" / "run_meta.json"
        self.eval_evo_ori_ape_path = self.eval_dir / "ori" / "evo_ape.zip"
        self.eval_evo_rec_ape_path = self.eval_dir / "rec" / "evo_ape.zip"


class _EvalRuntime:
    def __init__(self, root: Path):
        self.paths = _EvalPaths(root)
        self.exphub_root = root
        self.fps_arg = "1"

    def remove_in_exp(self, path):
        return None


def _touch_required_eval_inputs(paths: _EvalPaths) -> None:
    for directory in (paths.prepare_frames_dir, paths.decode_frames_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        paths.prepare_result_path,
        paths.encode_generation_units_path,
        paths.encode_prompts_path,
        paths.encode_result_path,
        paths.decode_calib_path,
        paths.decode_timestamps_path,
        paths.decode_report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def _infer_config() -> RunConfig:
    return RunConfig(
        dataset="dummy",
        sequence="seq",
        tag="tag",
        fps=1,
        start="0",
        dur="1",
        seed=12345,
        decode_profile="",
        log_level="quiet",
    )


class RuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_has_no_argparse_namespace_or_args_bag(self):
        plan = build_execution_plan(mode="infer", requested_step="encode", seed=12345)
        runtime = build_runtime(_infer_config(), plan)

        self.assertFalse(hasattr(runtime, "args"))
        self.assertNotIsInstance(runtime.config, argparse.Namespace)
        self.assertEqual(runtime.config.seed, 12345)
        self.assertEqual(runtime.execution_plan.stages, ("encode",))

    def test_runtime_does_not_carry_legacy_fields(self):
        plan = build_execution_plan(mode="train", requested_step=None, seed=12345)
        runtime = build_runtime(
            RunConfig(
                dataset="dummy",
                sequence="",
                tag="tag",
                fps=1,
                start="",
                dur="",
                seed=12345,
                decode_profile="",
                log_level="quiet",
            ),
            plan,
        )

        for field in (
            "keep_level",
            "lora_resume",
            "segment_policy",
            "gpus",
            "droid_repo",
            "droid_weights",
            "compression_benchmark",
        ):
            with self.subTest(field=field):
                self.assertFalse(hasattr(runtime.config, field))

    def test_standalone_eval_requires_prepared_gt_without_loading_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _EvalRuntime(Path(tmp))
            _touch_required_eval_inputs(runtime.paths)
            with mock.patch("exphub.eval.eval.get_platform_config") as get_platform_config:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "missing prepare/gt_traj.tum; rerun prepare or manually materialize the known GT artifact",
                ):
                    eval_pipeline.run(runtime)
            get_platform_config.assert_not_called()

    def test_partial_stage_run_dispatches_only_requested_stage(self):
        plan = build_execution_plan(mode="infer", requested_step="decode", seed=12345)
        runtime = build_runtime(_infer_config(), plan)
        fake_decode = mock.Mock()
        fake_decode.run.return_value = runtime.paths.decode_dir
        services = dict(runner_pipeline._SERVICE_BY_STAGE)
        services["decode"] = fake_decode

        with mock.patch("exphub.runner._validate_scripts_for_stages") as validate_scripts:
            with mock.patch("exphub.runner._SERVICE_BY_STAGE", services):
                with mock.patch("exphub.runner.write_run_start", return_value="start"):
                    with mock.patch("exphub.runner.update_run_status"):
                        run_runtime(runtime, plan)

        validate_scripts.assert_called_once_with(runtime, ("decode",))
        fake_decode.run.assert_called_once_with(runtime)

    def test_lora_does_not_stop_comfyui_pool(self):
        from exphub.lora import lora

        class Runtime:
            def __init__(self, root: Path):
                self.exphub_root = root
                self.execution_plan = build_execution_plan(mode="train", requested_step="lora", seed=12345)
                self.config = RunConfig(
                    dataset="dummy",
                    sequence="",
                    tag="tag",
                    fps=1,
                    start="",
                    dur="",
                    seed=12345,
                    decode_profile="",
                    log_level="quiet",
                )
                self.paths = type("Paths", (), {})()
                self.paths.trainset_dir = root / "trainset"
                self.paths.trainset_videos_dir = root / "trainset" / "videos"
                self.paths.trainset_metadata_path = root / "trainset" / "train_metadata.json"
                self.paths.trainset_stats_path = root / "trainset" / "stats.json"
                self.paths.lora_dir = root / "lora"
                self.paths.lora_config_path = root / "lora" / "lora_config.json"
                self.paths.lora_command_path = root / "lora" / "train_command.sh"
                self.paths.lora_log_path = root / "lora" / "lora.log"
                self.paths.lora_result_path = root / "lora" / "lora_result.json"

            def assert_under_exp(self, path):
                Path(path).resolve().relative_to(self.exphub_root.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = Runtime(root)
            runtime.paths.trainset_stats_path.parent.mkdir(parents=True, exist_ok=True)
            runtime.paths.trainset_stats_path.write_text("{}", encoding="utf-8")
            runtime.paths.trainset_metadata_path.write_text("{}", encoding="utf-8")
            repo = root / "repo"
            repo.mkdir()
            python = root / "python"
            python.write_text("", encoding="utf-8")
            with mock.patch("exphub.lora.lora._ensure_trainset", return_value=(root / "trainset", runtime.paths.trainset_metadata_path, runtime.paths.trainset_stats_path)):
                with mock.patch("exphub.lora.lora._load_profile", return_value=("default", {"cuda_visible_devices": "0", "script": "train.py"})):
                    with mock.patch("exphub.lora.lora._trainer_paths", return_value=(repo, repo / "train.py", python)):
                        with mock.patch("exphub.lora.lora._resolve_launcher", return_value=["python"]):
                            with mock.patch("exphub.lora.lora.run_cmd", return_value=0):
                                with mock.patch("exphub.lora.lora.subprocess.run") as subprocess_run:
                                    lora.run(runtime)
            subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
