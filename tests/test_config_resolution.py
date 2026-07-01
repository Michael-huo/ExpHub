from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from exphub.config import load_datasets_cfg
from exphub.decode.comfyui_client import resolve_comfyui_platform_config


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ConfigResolutionTests(unittest.TestCase):
    def test_static_decode_profiles_resolve_without_local_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_workflow = root / "base.json"
            lora_workflow = root / "lora.json"
            base_workflow.write_text("{}", encoding="utf-8")
            lora_workflow.write_text("{}", encoding="utf-8")
            platform_cfg = {
                "services": {
                    "comfyui": {
                        "active_profile": "base",
                        "instances": [
                            {
                                "name": "fixture",
                                "base_url": "http://127.0.0.1:1",
                                "output_root": str(root / "out"),
                            }
                        ],
                        "profiles": {
                            "base": {
                                "workflow_json": str(base_workflow),
                                "lora": {"enabled": False},
                                "nodes": {
                                    "ksampler": "3",
                                    "positive_prompt": "6",
                                    "negative_prompt": "7",
                                    "vae_decode": "8",
                                    "create_video": "57",
                                    "save_video": "58",
                                    "start_image": "70",
                                    "video_spec": "73",
                                    "end_image": "74",
                                    "save_image": "75",
                                },
                            },
                            "lora_gdut": {
                                "workflow_json": str(lora_workflow),
                                "lora": {
                                    "enabled": True,
                                    "name": "wan22_fun_5b_inp_gdut.safetensors",
                                    "strength_model": 1.0,
                                    "strength_clip": 1.0,
                                },
                                "nodes": {
                                    "ksampler": "11",
                                    "positive_prompt": "16",
                                    "negative_prompt": "12",
                                    "vae_decode": "6",
                                    "create_video": "7",
                                    "save_video": "8",
                                    "start_image": "4",
                                    "video_spec": "9",
                                    "end_image": "5",
                                    "save_image": "10",
                                    "lora_loader": "15",
                                },
                            },
                        },
                    }
                }
            }

            base = resolve_comfyui_platform_config(platform_cfg, exphub_root=root)
            lora = resolve_comfyui_platform_config(platform_cfg, exphub_root=root, decode_profile="lora_gdut")

            self.assertEqual(base["workflow_profile"].name, "base")
            self.assertFalse(base["workflow_profile"].lora_enabled)
            self.assertEqual(lora["workflow_profile"].name, "lora_gdut")
            self.assertTrue(lora["workflow_profile"].lora_enabled)

    def test_legacy_workflow_json_fallback_is_not_a_registered_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = Path(tmp) / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")
            platform_cfg = {
                "services": {
                    "comfyui": {
                        "workflow_json": str(workflow),
                        "base_url": "http://127.0.0.1:1",
                        "output_root": str(Path(tmp) / "out"),
                    }
                }
            }
            with self.assertRaisesRegex(RuntimeError, "profiles"):
                resolve_comfyui_platform_config(platform_cfg, exphub_root=tmp)

    def test_fixture_dataset_schemas_load_for_final_paper_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "datasets.json"
            payload = {"datasets": {}}
            for name in ("gdut", "tum", "ncd"):
                payload["datasets"][name] = {
                    "format": "rosbag",
                    "root": "datasets/{}".format(name),
                    "topic": "/camera",
                    "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "dist": []},
                    "sequences": {"seq": {"bag": "seq.bag"}},
                }
            _write_json(cfg_path, payload)

            loaded = load_datasets_cfg(cfg_path)
            self.assertEqual(set(loaded["datasets"]), {"gdut", "tum", "ncd"})


if __name__ == "__main__":
    unittest.main()
