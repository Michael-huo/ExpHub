from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from exphub.common.io import replace_nonempty_file, unique_sibling_temp_path
from exphub.common.paths import ExperimentPaths
from exphub.encode import motion_segment
from exphub.eval.summary_build import build_eval_summary
from exphub.execution_plan import build_execution_plan
from exphub.meta import ExperimentSpec
from exphub.prepare import prepare
from exphub.provenance import update_run_status, write_run_start
from exphub.runner import RunConfig


class _Paths:
    def __init__(self, root: Path):
        self.exp_dir = root
        self.run_meta_path = root / "run_meta.json"
        self.effective_config_path = root / "effective_config.yaml"
        self.command_path = root / "command.txt"
        self.git_state_path = root / "git_state.json"


class _Runtime:
    def __init__(self, root: Path):
        self.paths = _Paths(root / "run")
        self.exphub_root = root
        self.cfg_path = root / "config" / "datasets.json"
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg_path.write_text('{"datasets": {"dummy": {"root": "/data/dummy"}}}', encoding="utf-8")
        self.config = RunConfig(
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
        self.execution_plan = build_execution_plan(mode="infer", requested_step="prepare", seed=12345)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary_fixture(root: Path):
    exp_dir = root / "run"
    prepare = exp_dir / "prepare"
    encode = exp_dir / "encode"
    decode = exp_dir / "decode"
    eval_dir = exp_dir / "eval"
    for directory in (prepare / "raw_frames", encode, decode, eval_dir / "ori", eval_dir / "rec"):
        directory.mkdir(parents=True, exist_ok=True)
    (prepare / "raw_frames" / "000000.png").write_bytes(b"raw")
    _write_json(
        exp_dir / "run_meta.json",
        {
            "dataset": "dummy",
            "sequence": "seq",
            "tag": "tag",
            "fps": 1,
            "start": "0",
            "dur": "1",
            "seed": 12345,
            "decode_profile": "",
        },
    )
    _write_json(prepare / "prepare_result.json", {"ok": True})
    _write_json(encode / "generation_units.json", {"units": []})
    _write_json(encode / "prompts.json", {"prompts": []})
    _write_json(
        encode / "encode_result.json",
        {
            "profile": {"total_sec": 1.0},
            "raw_bytes": 300,
            "payload_bytes": 30,
            "payload_ratio": 0.1,
            "reduction_pct": 90,
            "raw_frame_count": 1,
            "transmitted_frame_count": 1,
            "generation_unit_count": 0,
            "unit_boundary_count": 0,
            "boundary_frame_bytes": 3,
            "json_payload_bytes": 27,
        },
    )
    _write_json(decode / "decode_report.json", {"total_runtime_sec": 2.0})
    _write_json(eval_dir / "ori" / "run_meta.json", {"runtime_sec": 1.0})
    _write_json(eval_dir / "rec" / "run_meta.json", {"runtime_sec": 1.0})
    return exp_dir


def _dataset_config(root: Path, sequence: str = "seq"):
    return SimpleNamespace(
        dataset="dummy",
        sequence=sequence,
        bag_path=str(root / "{}.bag".format(sequence)),
        topic="/camera",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "dist": []},
        image_size={"width": 2, "height": 2},
    )


def _sampled_frames():
    return SimpleNamespace(
        frames=["frame0"],
        start_sec=0.0,
        end_sec=1.0,
        dur_sec=1.0,
        prepared_to_source=[0],
        prepared_to_time_sec=[0.0],
        prepared_to_abs_time_sec=[10.0],
        prepared_to_ros_time_sec=[10.0],
    )


def _geometry():
    return SimpleNamespace(
        frames=["frame0"],
        original_resolution={"width": 2, "height": 2},
        normalized_resolution={"width": 32, "height": 32},
        original_intrinsics={"fx": 1.0},
        normalized_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "dist": []},
        transform_meta={},
    )


def _write_frames(_frames, frame_dir):
    path = Path(frame_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "000000.png").write_bytes(b"png")


def _copy_gt(_cfg, run_dir):
    path = Path(run_dir) / "gt_traj.tum"
    path.write_text("0 0 0 0 1 0 0 0\n", encoding="utf-8")
    return path


class ArtifactContractTests(unittest.TestCase):
    def _patch_prepare_dependencies(self, root: Path, sequence: str = "seq"):
        return mock.patch.multiple(
            prepare,
            load_dataset_config=mock.Mock(return_value=_dataset_config(root, sequence=sequence)),
            read_ros_frames=mock.Mock(return_value=["raw"]),
            sample_frames_to_target_fps=mock.Mock(return_value=_sampled_frames()),
            adapt_frames_and_intrinsics=mock.Mock(return_value=_geometry()),
            build_legal_grid=mock.Mock(return_value={}),
            maybe_write_frames=mock.Mock(side_effect=_write_frames),
            _copy_ground_truth_trajectory=mock.Mock(side_effect=_copy_gt),
        )

    def test_encode_decode_helpers_use_final_root_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ExperimentSpec(
                exphub_root=Path(tmp),
                mode="infer",
                dataset="gdut",
                sequence="seq",
                tag="tag",
                start="0",
                dur="1",
                fps=24,
            )
            paths = ExperimentPaths.from_spec(spec)

            self.assertEqual(paths.encode_generation_units_path, paths.encode_dir / "generation_units.json")
            self.assertEqual(paths.encode_prompts_path, paths.encode_dir / "prompts.json")
            self.assertEqual(paths.encode_motion_segments_path, paths.encode_dir / "motion_segments.json")
            self.assertEqual(paths.encode_motion_overview_path, paths.encode_dir / "motion_overview.png")
            self.assertEqual(paths.encode_motion_benchmark_canonical_json_path, paths.encode_dir / "motion_benchmark" / "summary.json")
            self.assertEqual(paths.encode_compression_report_path, paths.encode_dir / "compression_benchmark" / "report.json")

            self.assertEqual(paths.decode_runs_dir, paths.decode_dir / ".comfyui_tmp")
            self.assertEqual(paths.decode_calib_path, paths.decode_dir / "calib.txt")
            self.assertEqual(paths.decode_timestamps_path, paths.decode_dir / "timestamps.txt")
            self.assertEqual(paths.decode_frames_dir, paths.decode_dir / "reconstructed_frames")
            self.assertEqual(paths.decode_image_quality_canonical_json_path, paths.decode_dir / "image_quality" / "summary.json")
            self.assertEqual(paths.decode_compression_report_path, paths.decode_dir / "compression_benchmark" / "report.json")

            self.assertEqual(paths.eval_compression_summary_path, paths.eval_dir / "compression_benchmark" / "summary.json")

    def test_motion_overview_writes_final_root_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "frames"
            frame_dir.mkdir(parents=True)

            import cv2

            frames = []
            for idx in range(3):
                image = motion_segment.np.full((32, 32, 3), 48 + idx * 40, dtype=motion_segment.np.uint8)
                path = frame_dir / "{:06d}.png".format(idx)
                self.assertTrue(cv2.imwrite(str(path), image))
                frames.append(path)

            grays = [motion_segment._read_gray(path) for path in frames]
            out_path = root / "encode" / "motion_segments.json"
            result = motion_segment._write_motion_overview(
                out_path,
                frames,
                grays,
                pair_states=[
                    {"steering_response": 0.1},
                    {"steering_response": 0.3},
                ],
                motion_states=[
                    {"start_idx": 0, "end_idx": 2, "motion_label": "forward"},
                ],
                num_frames=3,
            )

            final_path = root / "encode" / "motion_overview.png"
            self.assertEqual(result, "motion_overview.png")
            self.assertTrue(final_path.is_file())
            self.assertGreater(final_path.stat().st_size, 0)
            self.assertFalse((root / "encode" / "diagnostics" / "motion_overview.png").exists())

    def test_infer_prepare_writes_raw_frames_and_removes_stale_frames_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_dir = root / "prepare"
            stale = prepare_dir / "frames"
            stale.mkdir(parents=True)
            (stale / "old.png").write_bytes(b"old")
            raw_frames = prepare_dir / "raw_frames"

            with self._patch_prepare_dependencies(root):
                result = prepare._run_single_prepare(
                    mode="infer",
                    config_path=root / "config" / "datasets.json",
                    dataset_name="dummy",
                    sequence_name="seq",
                    target_fps=24,
                    start_sec=0,
                    dur_sec=1,
                    output_dir=prepare_dir,
                    frame_dir=raw_frames,
                )

            self.assertEqual(Path(result.frame_dir), raw_frames.resolve())
            self.assertTrue((raw_frames / "000000.png").is_file())
            self.assertFalse(stale.exists())
            payload = json.loads((prepare_dir / "prepare_result.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["frame_dir"]), raw_frames.resolve())

    def test_runtime_infer_prepare_uses_prepare_raw_frames_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "run"
            prepare_dir = exp_dir / "prepare"
            raw_frames = prepare_dir / "raw_frames"
            stale = prepare_dir / "frames"
            stale.mkdir(parents=True)
            (stale / "old.png").write_bytes(b"old")

            class Runtime:
                cfg_path = root / "config" / "datasets.json"
                execution_plan = SimpleNamespace(mode="infer")
                spec = SimpleNamespace(
                    dataset="dummy",
                    sequence="seq",
                    fps=24,
                    start=0,
                    dur=1,
                    exp_name="run",
                )
                paths = SimpleNamespace(
                    exp_dir=exp_dir,
                    prepare_dir=prepare_dir,
                    prepare_frames_dir=raw_frames,
                )

                def ensure_clean_exp_dir(self):
                    self.paths.exp_dir.mkdir(parents=True, exist_ok=True)

            with self._patch_prepare_dependencies(root):
                out_dir = prepare.run(Runtime())

            self.assertEqual(out_dir, prepare_dir)
            self.assertTrue((raw_frames / "000000.png").is_file())
            self.assertFalse(stale.exists())
            payload = json.loads((prepare_dir / "prepare_result.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["frame_dir"]), raw_frames.resolve())

    def test_train_prepare_sequence_keeps_sequence_frames_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "prepare" / "sequences" / "seq"
            frames = sequence_dir / "frames"

            with self._patch_prepare_dependencies(root):
                result = prepare._run_single_prepare(
                    mode="train",
                    config_path=root / "config" / "datasets.json",
                    dataset_name="dummy",
                    sequence_name="seq",
                    target_fps=24,
                    output_dir=sequence_dir,
                    frame_dir=frames,
                )

            self.assertEqual(Path(result.frame_dir), frames.resolve())
            self.assertTrue((frames / "000000.png").is_file())
            self.assertFalse((sequence_dir / "raw_frames").exists())

    def test_root_provenance_schema_is_minimal_and_lifecycle_updates_only_status_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _Runtime(Path(tmp))
            start_time = write_run_start(runtime, ["--mode", "infer", "--step", "prepare"])

            run_meta = json.loads(runtime.paths.run_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(run_meta["status"], "running")
            self.assertNotIn("artifact_paths", run_meta)
            self.assertEqual(
                set(json.loads(runtime.paths.git_state_path.read_text(encoding="utf-8"))),
                {"commit", "branch", "worktree_dirty"},
            )
            self.assertIn("python3 -m exphub --mode infer --step prepare", runtime.paths.command_path.read_text(encoding="utf-8"))
            self.assertIn("dataset_config", runtime.paths.effective_config_path.read_text(encoding="utf-8"))

            update_run_status(runtime, status="failed", start_time=start_time, error="boom\ntrace")
            failed_meta = json.loads(runtime.paths.run_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(failed_meta["status"], "failed")
            self.assertEqual(failed_meta["error"], "boom trace")
            self.assertNotIn("changed_files", json.loads(runtime.paths.git_state_path.read_text(encoding="utf-8")))

    def test_fresh_required_artifact_replacement_preserves_old_final_until_temp_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "preview.mp4"
            final.write_text("old", encoding="utf-8")

            empty = unique_sibling_temp_path(final)
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                replace_nonempty_file(empty, final, "preview")
            self.assertEqual(final.read_text(encoding="utf-8"), "old")

            fresh = unique_sibling_temp_path(final)
            fresh.write_text("new", encoding="utf-8")
            replace_nonempty_file(fresh, final, "preview")
            self.assertEqual(final.read_text(encoding="utf-8"), "new")

    def test_eval_summary_partial_and_full_runtime_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = _summary_fixture(Path(tmp))
            config = {
                "exp_dir": str(exp_dir),
                "out_dir": str(exp_dir / "eval"),
                "prepare_result": str(exp_dir / "prepare" / "prepare_result.json"),
                "generation_units": str(exp_dir / "encode" / "generation_units.json"),
                "prompts": str(exp_dir / "encode" / "prompts.json"),
                "encode_result": str(exp_dir / "encode" / "encode_result.json"),
                "prepare_frames_dir": str(exp_dir / "prepare" / "raw_frames"),
                "decode_report": str(exp_dir / "decode" / "decode_report.json"),
                "ori_run_meta": str(exp_dir / "eval" / "ori" / "run_meta.json"),
                "rec_run_meta": str(exp_dir / "eval" / "rec" / "run_meta.json"),
                "evo_result": {"ori_ape_rmse": 0.1, "rec_ape_rmse": 0.2},
                "eval_runtime_sec": 4.0,
            }
            build_eval_summary(dict(config, complete_main_chain=False, stage_times={"eval": 4.0}))
            partial = json.loads((exp_dir / "eval" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(partial["runtime"]["eval_time_s"], 4.0)
            self.assertIsNone(partial["runtime"]["main_pipeline_time_s"])
            self.assertIsNone(partial["runtime"]["prepare_time_s"])
            self.assertIn("ratio", partial["payload"])
            self.assertIn("raw_frame_count", partial["payload"])
            self.assertIn("ape_delta_rec_minus_ori_m", partial["vslam"])
            self.assertNotIn("ori_" + "r" + "pe" + "_trans_m", partial["vslam"])
            self.assertFalse((exp_dir / "eval" / ("eval_" + "summary.txt")).exists())
            self.assertFalse((exp_dir / "eval" / ("eval_" + "details.csv")).exists())
            self.assertFalse((exp_dir / "eval" / ("eval_" + "compression_report.json")).exists())

            build_eval_summary(
                dict(
                    config,
                    complete_main_chain=True,
                    stage_times={"prepare": 1.0, "encode": 2.0, "decode": 3.0},
                    eval_runtime_sec=4.0,
                    main_pipeline_wall_time_s=9.5,
                )
            )
            full = json.loads((exp_dir / "eval" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(full["runtime"]["prepare_time_s"], 1.0)
            self.assertEqual(full["runtime"]["main_pipeline_time_s"], 9.5)


if __name__ == "__main__":
    unittest.main()
