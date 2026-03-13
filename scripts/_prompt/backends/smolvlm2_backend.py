from __future__ import annotations

import time
from pathlib import Path

from .base import PromptBackend


DEFAULT_SMOLVLM2_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

def _torch_dtype_from_name(torch_mod, dtype_name):
    # type: (object, str) -> object
    name = str(dtype_name or "bfloat16").strip().lower()
    if name == "bfloat16":
        return torch_mod.bfloat16
    if name == "float16":
        return torch_mod.float16
    raise SystemExit("[ERR] unsupported prompt dtype: {}".format(dtype_name))


class SmolVlm2PromptBackend(PromptBackend):
    name = "smolvlm2"

    def __init__(
        self,
        model_ref="",  # type: str
        dtype="bfloat16",  # type: str
        max_new_tokens=48,  # type: int
    ):
        self.model_ref = str(model_ref or DEFAULT_SMOLVLM2_MODEL_ID).strip() or DEFAULT_SMOLVLM2_MODEL_ID
        self.requested_dtype = str(dtype or "bfloat16").strip().lower()
        self.max_new_tokens = int(max_new_tokens)
        self.processor = None
        self.model = None
        self._torch = None
        self._pil_image = None
        self._load_meta = {
            "backend": self.name,
            "model_dir": "",
            "model_id": self.model_ref,
            "attn_impl": "sdpa",
            "dtype": self.requested_dtype,
            "processor_load_sec": 0.0,
            "model_load_sec": 0.0,
        }

    def load(self):
        # type: () -> None
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise SystemExit("[ERR] failed to import SmolVLM2 prompt backend dependencies: {}".format(exc))

        self._torch = torch
        self._pil_image = Image
        torch_dtype = _torch_dtype_from_name(torch, self.requested_dtype)

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_ref)
        self._load_meta["processor_load_sec"] = float(time.time() - t0)

        t1 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_ref,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self._load_meta["model_load_sec"] = float(time.time() - t1)

        model_path = Path(self.model_ref).expanduser()
        if model_path.exists():
            self._load_meta["model_dir"] = str(model_path.resolve())
            self._load_meta["model_id"] = ""

    def _load_images(self, image_paths):
        # type: (list) -> list
        images = []
        for image_path in image_paths:
            path_obj = Path(image_path).resolve()
            with self._pil_image.open(str(path_obj)) as img:
                images.append(img.convert("RGB").copy())
        return images

    def _build_inputs(self, prompt_text, images):
        attempts = [
            {"text": [prompt_text], "images": images},
            {"text": prompt_text, "images": images},
            {"text": [prompt_text], "images": [images]},
            {"text": prompt_text, "images": [images]},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return self.processor(padding=True, return_tensors="pt", **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError("SmolVLM2 processor could not encode multi-image input: {}".format(last_error))

    def generate(self, image_paths, instruction):
        # type: (list, str) -> str
        if self.processor is None or self.model is None:
            raise RuntimeError("SmolVlm2PromptBackend.load() must be called before generate()")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in image_paths] + [{"type": "text", "text": instruction}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = self._load_images(image_paths)
        inputs = self._build_inputs(prompt_text, images).to(self.model.device)

        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[1] :]
        prompt = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return str(prompt or "")

    def meta(self):
        # type: () -> dict
        return dict(self._load_meta)
