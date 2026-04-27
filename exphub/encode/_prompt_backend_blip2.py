from __future__ import annotations

import argparse
import re
from pathlib import Path

from exphub.common.io import read_json_dict, write_json_atomic


DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _collapse_ws(value):
    return " ".join(str(value or "").strip().split()).strip()


def _clean_caption(value):
    text = _collapse_ws(value)
    text = re.sub(r"^(question|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(question|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _load_image(image_cls, frame_path):
    path = Path(frame_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("BLIP-2 input image not found: {}".format(path))
    with image_cls.open(str(path)) as image_obj:
        return image_obj.convert("RGB").copy()


def _move_inputs(inputs, torch_module, device):
    moved = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            if torch_module.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=torch_module.float16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def generate_captions(input_json, output_json, model_ref, device, max_new_tokens, num_beams):
    try:
        import torch
        from PIL import Image
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except Exception as exc:
        raise RuntimeError("failed to import BLIP-2 backend dependencies: {}".format(exc)) from exc

    payload = read_json_dict(input_json)
    items = list(payload.get("items") or [])
    instruction = _collapse_ws(payload.get("instruction")) or "Briefly describe the visible scene."
    model_name = str(model_ref or DEFAULT_BLIP2_MODEL).strip() or DEFAULT_BLIP2_MODEL
    device_name = str(device or "cuda:0").strip() or "cuda:0"

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BLIP-2 requested CUDA device {} but torch.cuda.is_available() is false".format(device_name))

    try:
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device_name)
        model.eval()
    except Exception as exc:
        raise RuntimeError("failed to load BLIP-2 model {}: {}".format(model_name, exc)) from exc

    out_items = []
    for idx, raw_item in enumerate(items):
        item = _as_dict(raw_item)
        frame_path = str(item.get("frame_path", "") or "")
        if not frame_path:
            raise RuntimeError("BLIP-2 input item {} missing frame_path".format(idx))
        image = _load_image(Image, frame_path)
        try:
            inputs = processor(images=image, return_tensors="pt")
            inputs = _move_inputs(inputs, torch, device_name)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    do_sample=False,
                )
            input_ids = inputs.get("input_ids")
            if input_ids is not None and generated.shape[-1] > input_ids.shape[-1]:
                generated = generated[:, input_ids.shape[-1] :]
            caption = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
        except Exception as exc:
            raise RuntimeError("BLIP-2 caption failed for {}: {}".format(frame_path, exc)) from exc
        out_item = dict(item)
        out_item["caption"] = _clean_caption(caption)
        out_items.append(out_item)

    write_json_atomic(
        output_json,
        {
            "backend": "blip2",
            "model": model_name,
            "instruction": instruction,
            "items": out_items,
        },
        indent=2,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m exphub.encode._prompt_backend_blip2")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default=DEFAULT_BLIP2_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generate_captions(
        input_json=args.input_json,
        output_json=args.output_json,
        model_ref=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
