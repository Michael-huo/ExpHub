from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from exphub import experiments
from exphub.encode import encode as encode_pipeline
from exphub.execution_plan import build_execution_plan


class _Paths:
    def __init__(self, root: Path):
        self.exp_dir = root
        self.prepare_dir = root / "prepare"
        self.prepare_result_path = self.prepare_dir / "prepare_result.json"
        self.prepare_frames_dir = self.prepare_dir / "raw_frames"
        self.prepare_gt_traj_path = self.prepare_dir / "gt_traj.tum"
        self.encode_dir = root / "encode"
        self.encode_result_path = self.encode_dir / "encode_result.json"
        self.encode_generation_units_path = self.encode_dir / "generation_units.json"
        self.encode_prompts_path = self.encode_dir / "prompts.json"
        self.encode_motion_segments_path = self.encode_dir / "motion_segments.json"
        self.encode_semantic_anchors_path = self.encode_dir / "semantic_anchors.json"
        self.encode_motion_overview_path = self.encode_dir / "motion_overview.png"
        self.encode_compression_dir = self.encode_dir / "compression_benchmark"
        self.encode_compression_report_path = self.encode_compression_dir / "report.json"
        self.encode_motion_benchmark_dir = self.encode_dir / "motion_benchmark"
        self.encode_motion_benchmark_report_path = self.encode_motion_benchmark_dir / "report.json"
        self.encode_motion_benchmark_pairs_csv_path = self.encode_motion_benchmark_dir / "pairs.csv"
        self.encode_motion_benchmark_overview_path = self.encode_motion_benchmark_dir / "overview.png"
        self.encode_motion_benchmark_canonical_json_path = self.encode_motion_benchmark_dir / "summary.json"
        self.encode_motion_benchmark_canonical_csv_path = self.encode_motion_benchmark_dir / "summary.csv"
        self.decode_dir = root / "decode"
        self.decode_frames_dir = self.decode_dir / "reconstructed_frames"
        self.decode_report_path = self.decode_dir / "decode_report.json"
        self.decode_calib_path = self.decode_dir / "calib.txt"
        self.decode_timestamps_path = self.decode_dir / "timestamps.txt"
        self.decode_compression_dir = self.decode_dir / "compression_benchmark"
        self.decode_compression_report_path = self.decode_compression_dir / "report.json"
        self.decode_image_quality_dir = self.decode_dir / "image_quality"
        self.decode_image_quality_report_path = self.decode_image_quality_dir / "report.json"
        self.decode_image_quality_details_path = self.decode_image_quality_dir / "details.csv"
        self.decode_image_quality_canonical_json_path = self.decode_image_quality_dir / "summary.json"
        self.decode_image_quality_canonical_csv_path = self.decode_image_quality_dir / "summary.csv"
        self.eval_dir = root / "eval"
        self.eval_canonical_summary_path = self.eval_dir / "summary.json"
        self.eval_compression_dir = self.eval_dir / "compression_benchmark"
        self.eval_compression_summary_path = self.eval_compression_dir / "summary.json"
        self.eval_compression_summary_csv_path = self.eval_compression_dir / "summary.csv"


class _Runtime:
    def __init__(self, root: Path):
        self.paths = _Paths(root)
        self.exphub_root = root
        self.spec = type("Spec", (), {"fps": 1.0, "dur": "1"})()
        self.step_runner = mock.Mock()

    def remove_in_exp(self, path):
        return None


def _prepare_runtime(root: Path) -> _Runtime:
    runtime = _Runtime(root)
    for directory in (
        runtime.paths.prepare_frames_dir,
        runtime.paths.encode_dir / "hvm_payload",
        runtime.paths.decode_frames_dir,
        runtime.paths.eval_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runtime.paths.prepare_result_path.write_text('{"ok": true}', encoding="utf-8")
    runtime.paths.prepare_gt_traj_path.write_text("0 0 0 0 1 0 0 0\n", encoding="utf-8")
    runtime.paths.encode_result_path.write_text(
        '{"profile": {"total_sec": 1.5}, "raw_bytes": 100, "payload_bytes": 10, "reduction_pct": 90, "transmitted_frame_count": 1, "generation_unit_count": 1}',
        encoding="utf-8",
    )
    runtime.paths.encode_generation_units_path.write_text("{}", encoding="utf-8")
    runtime.paths.encode_prompts_path.write_text("{}", encoding="utf-8")
    runtime.paths.decode_report_path.write_text("{}", encoding="utf-8")
    runtime.paths.decode_calib_path.write_text("", encoding="utf-8")
    runtime.paths.decode_timestamps_path.write_text("", encoding="utf-8")
    runtime.paths.eval_canonical_summary_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.paths.eval_canonical_summary_path.write_text('{"vslam": {"ori_ape_rmse_m": 0.1, "rec_ape_rmse_m": 0.2}}', encoding="utf-8")
    return runtime


class ExperimentDispatchTests(unittest.TestCase):
    def test_motion_benchmark_helper_does_not_receive_or_patch_main_encode_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _prepare_runtime(Path(tmp))
            def motion_side_effect(*_args, **_kwargs):
                runtime.paths.encode_motion_benchmark_dir.mkdir(parents=True, exist_ok=True)
                for path in (
                    runtime.paths.encode_motion_benchmark_report_path,
                    runtime.paths.encode_motion_benchmark_pairs_csv_path,
                    runtime.paths.encode_motion_benchmark_overview_path,
                    runtime.paths.encode_motion_benchmark_canonical_json_path,
                    runtime.paths.encode_motion_benchmark_canonical_csv_path,
                ):
                    path.write_text("", encoding="utf-8")
                return {}

            with mock.patch("exphub.encode.encode._patch_encode_result_profile") as patch_profile:
                with mock.patch("exphub.encode.encode.run_motion_benchmark", side_effect=motion_side_effect) as motion_benchmark:
                    encode_pipeline.run_motion_benchmark_extra(runtime)

            patch_profile.assert_not_called()
            motion_benchmark.assert_called_once()
            kwargs = motion_benchmark.call_args.kwargs
            self.assertNotIn("encode_result", kwargs)
            self.assertNotIn("encode_result_path", kwargs)
            self.assertFalse((Path(tmp) / "encode" / "summary.json").exists())
            self.assertFalse((Path(tmp) / "encode" / "summary.csv").exists())
            self.assertFalse((Path(tmp) / "encode" / ("motion_benchmark_" + "summary.json")).exists())
            self.assertFalse((Path(tmp) / "encode" / ("motion_benchmark_" + "summary.csv")).exists())
            self.assertTrue((Path(tmp) / "encode" / "motion_benchmark" / "summary.json").is_file())
            self.assertTrue((Path(tmp) / "encode" / "motion_benchmark" / "summary.csv").is_file())

    def test_compression_benchmark_dispatch_writes_existing_benchmark_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _prepare_runtime(Path(tmp))
            plan = build_execution_plan(
                mode="infer",
                requested_step=None,
                experiments=("compression-benchmark",),
                seed=12345,
            )
            with mock.patch("exphub.experiments.get_platform_config", return_value={"repos": {}, "models": {}}):
                with mock.patch("exphub.experiments.encode_pipeline.run_compression_benchmark_encode_extra") as enc:
                    with mock.patch("exphub.experiments.decode_pipeline.run_compression_benchmark_decode_extra") as dec:
                        def eval_side_effect(*_args, **_kwargs):
                            runtime.paths.eval_compression_dir.mkdir(parents=True, exist_ok=True)
                            runtime.paths.eval_compression_summary_path.write_text("{}", encoding="utf-8")
                            runtime.paths.eval_compression_summary_csv_path.write_text("method\n", encoding="utf-8")
                            return {}

                        runtime.step_runner.run_env_python.side_effect = eval_side_effect
                        with redirect_stdout(io.StringIO()):
                            experiments.run_requested_experiments(runtime, plan)

            enc.assert_called_once_with(runtime)
            dec.assert_called_once_with(runtime)
            runtime.step_runner.run_env_python.assert_called_once()
            cmd = runtime.step_runner.run_env_python.call_args.args[0]
            kwargs = runtime.step_runner.run_env_python.call_args.kwargs
            self.assertEqual(kwargs["phase_name"], "slam")
            self.assertEqual(kwargs["log_name"], "compression_benchmark_eval.log")
            self.assertEqual(kwargs["cwd"], runtime.exphub_root)
            self.assertIn("exphub.eval.compression_benchmark", cmd)
            self.assertIn("--run-compression-benchmark-eval", cmd)
            self.assertIn(str(runtime.paths.eval_compression_dir), cmd)
            self.assertIn(str(runtime.paths.decode_compression_report_path), cmd)
            benchmark_files = {path.name for path in (Path(tmp) / "eval" / "compression_benchmark").iterdir()}
            self.assertEqual(benchmark_files, {"summary.json", "summary.csv"})

    def test_image_quality_dispatch_uses_existing_decode_quality_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _prepare_runtime(Path(tmp))
            plan = build_execution_plan(
                mode="infer",
                requested_step=None,
                experiments=("image-quality",),
                seed=12345,
            )
            def image_quality_side_effect(runtime_arg, **_kwargs):
                runtime_arg.paths.decode_image_quality_dir.mkdir(parents=True, exist_ok=True)
                runtime_arg.paths.decode_image_quality_canonical_json_path.write_text("{}", encoding="utf-8")
                runtime_arg.paths.decode_image_quality_canonical_csv_path.write_text("metric\n", encoding="utf-8")
                return {}

            with mock.patch("exphub.experiments.run_decode_image_quality_subprocess", side_effect=image_quality_side_effect) as image_quality:
                with redirect_stdout(io.StringIO()):
                    experiments.run_requested_experiments(runtime, plan)

            image_quality.assert_called_once_with(
                runtime,
                stride=experiments.IMAGE_QUALITY_STRIDE,
                max_frames=experiments.IMAGE_QUALITY_MAX_FRAMES,
                device=experiments.IMAGE_QUALITY_DEVICE,
            )
            self.assertFalse((Path(tmp) / "decode" / "summary.json").exists())
            self.assertFalse((Path(tmp) / "decode" / "summary.csv").exists())
            self.assertFalse((Path(tmp) / "decode" / ("image_quality_" + "summary.txt")).exists())
            self.assertFalse((Path(tmp) / "decode" / ("image_quality_" + "summary.json")).exists())
            self.assertFalse((Path(tmp) / "decode" / ("image_quality_" + "summary.csv")).exists())
            self.assertTrue((Path(tmp) / "decode" / "image_quality" / "summary.json").is_file())
            self.assertTrue((Path(tmp) / "decode" / "image_quality" / "summary.csv").is_file())

    def test_experiment_dispatch_logs_only_basic_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _prepare_runtime(Path(tmp))
            plan = build_execution_plan(
                mode="infer",
                requested_step=None,
                experiments=("motion-benchmark", "compression-benchmark", "image-quality"),
                seed=12345,
            )

            def motion_side_effect(runtime_arg):
                runtime_arg.paths.encode_motion_benchmark_dir.mkdir(parents=True, exist_ok=True)
                runtime_arg.paths.encode_motion_benchmark_canonical_json_path.write_text("{}", encoding="utf-8")
                runtime_arg.paths.encode_motion_benchmark_canonical_csv_path.write_text("method\n", encoding="utf-8")

            def eval_side_effect(*_args, **_kwargs):
                runtime.paths.eval_compression_dir.mkdir(parents=True, exist_ok=True)
                runtime.paths.eval_compression_summary_path.write_text("{}", encoding="utf-8")
                runtime.paths.eval_compression_summary_csv_path.write_text("method\n", encoding="utf-8")

            def image_quality_side_effect(runtime_arg, **_kwargs):
                runtime_arg.paths.decode_image_quality_dir.mkdir(parents=True, exist_ok=True)
                runtime_arg.paths.decode_image_quality_canonical_json_path.write_text("{}", encoding="utf-8")
                runtime_arg.paths.decode_image_quality_canonical_csv_path.write_text("metric\n", encoding="utf-8")

            output = io.StringIO()
            with mock.patch("exphub.experiments.get_platform_config", return_value={"repos": {}, "models": {}}):
                with mock.patch("exphub.experiments.encode_pipeline.run_motion_benchmark_extra", side_effect=motion_side_effect):
                    with mock.patch("exphub.experiments.encode_pipeline.run_compression_benchmark_encode_extra"):
                        with mock.patch("exphub.experiments.decode_pipeline.run_compression_benchmark_decode_extra"):
                            runtime.step_runner.run_env_python.side_effect = eval_side_effect
                            with mock.patch(
                                "exphub.experiments.run_decode_image_quality_subprocess",
                                side_effect=image_quality_side_effect,
                            ):
                                with redirect_stdout(output):
                                    experiments.run_requested_experiments(runtime, plan)

            text = output.getvalue()
            self.assertIn("experiment motion-benchmark start", text)
            self.assertIn("experiment image-quality done", text)
            self.assertNotIn("phase_correlation." + "runtime_sec", text)
            self.assertNotIn("[Compression Benchmark: encode] " + "Method " + "Summary", text)
            self.assertNotIn("[Compression Benchmark: decode] " + "Method " + "Summary", text)
            self.assertNotIn("[Compression Benchmark: eval] " + "Method " + "Summary", text)
            self.assertNotIn("matched_" + "frames", text)

    def test_normal_encode_motion_overview_is_not_gated_by_motion_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _prepare_runtime(Path(tmp))
            paths = type("EncodePaths", (), {})()
            paths.exp_dir = runtime.paths.exp_dir
            paths.prepare_result_path = runtime.paths.prepare_result_path
            paths.prepare_frames_dir = runtime.paths.prepare_frames_dir
            paths.encode_dir = runtime.paths.encode_dir
            paths.encode_motion_segments_path = runtime.paths.encode_motion_segments_path
            paths.encode_semantic_anchors_path = runtime.paths.encode_semantic_anchors_path
            paths.encode_generation_units_path = runtime.paths.encode_generation_units_path
            paths.encode_prompts_path = runtime.paths.encode_prompts_path
            paths.encode_result_path = runtime.paths.encode_result_path
            paths.encode_motion_overview_path = runtime.paths.encode_motion_overview_path

            def write_semantic(*_args, **_kwargs):
                paths.encode_semantic_anchors_path.write_text('{"anchors": []}', encoding="utf-8")

            runtime.step_runner = mock.Mock()
            runtime.step_runner.run_env_python.side_effect = write_semantic
            def build_motion_side_effect(*_args, **_kwargs):
                paths.encode_motion_segments_path.parent.mkdir(parents=True, exist_ok=True)
                paths.encode_motion_segments_path.write_text('{"segments": []}', encoding="utf-8")
                paths.encode_motion_overview_path.write_text("", encoding="utf-8")
                return {"segments": []}

            def build_units_side_effect(*_args, **_kwargs):
                paths.encode_generation_units_path.parent.mkdir(parents=True, exist_ok=True)
                paths.encode_generation_units_path.write_text('{"units": []}', encoding="utf-8")
                return {"units": []}

            def build_prompts_side_effect(*_args, **_kwargs):
                paths.encode_prompts_path.parent.mkdir(parents=True, exist_ok=True)
                paths.encode_prompts_path.write_text('{"units": []}', encoding="utf-8")
                return {"units": []}

            def write_outputs_side_effect(*_args, **_kwargs):
                paths.encode_result_path.write_text(
                    '{"profile": {}, "raw_bytes": 1, "payload_bytes": 1, "reduction_pct": 0, "transmitted_frame_count": 1, "generation_unit_count": 1}',
                    encoding="utf-8",
                )
                return paths.encode_result_path

            with mock.patch("exphub.encode.encode.build_motion_segments", side_effect=build_motion_side_effect) as build_motion:
                with mock.patch("exphub.encode.encode.build_generation_units", side_effect=build_units_side_effect):
                    with mock.patch("exphub.encode.encode.build_prompts", side_effect=build_prompts_side_effect):
                        with mock.patch("exphub.encode.encode.write_encode_outputs", side_effect=write_outputs_side_effect):
                            with mock.patch("exphub.encode.encode.run_motion_benchmark") as motion_benchmark:
                                with mock.patch("exphub.encode.encode._run_infer_payload_hooks"):
                                    encode_pipeline._run_single_encode(runtime, paths)
            motion_overview_exists = (Path(tmp) / "encode" / "motion_overview.png").is_file()

        build_motion.assert_called_once()
        motion_benchmark.assert_not_called()
        self.assertTrue(motion_overview_exists)


if __name__ == "__main__":
    unittest.main()
