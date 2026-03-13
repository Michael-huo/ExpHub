from __future__ import annotations

import copy
import os
import time
from pathlib import Path

from .base import PromptBackend


class QwenPromptBackend(PromptBackend):
    name = "qwen"

    def __init__(
        self,
        model_ref="",  # type: str
        use_fast=False,  # type: bool
        min_pixels=0,  # type: int
        max_pixels=0,  # type: int
        max_new_tokens=80,  # type: int
    ):
        self.model_ref = str(model_ref or "").strip()
        self.use_fast = bool(use_fast)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.processor = None
        self.model = None
        self._process_vision_info = None
        self._load_meta = {
            "backend": self.name,
            "model_dir": self.model_ref,
            "model_id": "",
            "attn_impl": "auto",
            "dtype": "auto",
            "processor_load_sec": 0.0,
            "model_load_sec": 0.0,
        }

    def load(self):
        # type: () -> None
        if not self.model_ref:
            raise SystemExit(
                "[ERR] --model_dir is empty. Set models.qwen2_vl.path in config/platform.yaml or pass --model_dir."
            )

        model_dir = Path(self.model_ref).expanduser().resolve()
        if not model_dir.is_dir():
            raise SystemExit(
                "[ERR] model_dir not found or not a directory: {}. "
                "Please set --model_dir to a valid Qwen2-VL model directory.".format(model_dir)
            )

        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            raise SystemExit("[ERR] failed to import Qwen prompt backend dependencies: {}".format(exc))

        self._torch = torch
        self._process_vision_info = process_vision_info

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            use_fast=self.use_fast,
        )
        self._load_meta["processor_load_sec"] = float(time.time() - t0)

        t1 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        self._load_meta["model_load_sec"] = float(time.time() - t1)
        self._load_meta["model_dir"] = str(model_dir)

        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        self.generation_config.num_beams = 1
        self.generation_config.temperature = 1.0
        self.generation_config.top_p = 1.0
        self.generation_config.top_k = 50
        setattr(self.generation_config, "max_new_tokens", self.max_new_tokens)

    def generate(self, image_paths, instruction):
        # type: (list, str) -> str
        if self.processor is None or self.model is None:
            raise RuntimeError("QwenPromptBackend.load() must be called before generate()")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": "file://{}".format(str(Path(p).resolve()))} for p in image_paths]
                + [{"type": "text", "text": instruction}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        prompt = self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return str(prompt or "")

    def meta(self):
        # type: () -> dict
        return dict(self._load_meta)
