from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from ultralytics.nn.tasks import YOLOEModel as UltralyticsYOLOEModel
from ultralytics.nn.tasks import YOLOESegModel as UltralyticsYOLOESegModel
from ultralytics.utils.downloads import attempt_download_asset

ASSET_PREFIX = "__asset__."
DEFAULT_SOURCE_MODEL = Path("model-bin/openvision/yoloe26-x-seg/model.pt")
DEFAULT_OUTPUT_ROOT = Path("model-bin/MigoXV")
DEFAULT_MODEL_NAME = "yoloe26-x-seg"
DEFAULT_TEXT_ENCODER_NAME = "mobileclip2-b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an OpenVision YOLOE checkpoint into split safetensors dirs.")
    parser.add_argument("--source-model", default=str(DEFAULT_SOURCE_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--text-encoder-name", default=DEFAULT_TEXT_ENCODER_NAME)
    return parser.parse_args()


def load_source_model(source_model: Path) -> tuple[dict, torch.nn.Module]:
    checkpoint = torch.load(source_model, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
    else:
        model = checkpoint
        checkpoint = {}
    if not isinstance(model, (UltralyticsYOLOEModel, UltralyticsYOLOESegModel)):
        raise TypeError(f"Unsupported checkpoint model type: {type(model)!r}")
    return checkpoint, model.eval()


def infer_visual_dtype(model: torch.nn.Module) -> str:
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            return str(tensor.dtype).removeprefix("torch.")
    return "float32"


def resolve_text_encoder_asset(text_model_variant: str) -> tuple[str, str]:
    base, _, size = str(text_model_variant).partition(":")
    if base == "mobileclip2":
        return "mobileclip2", "mobileclip2_b.ts"
    if base == "mobileclip":
        if size in {"", "blt"}:
            return "mobileclip", "mobileclip_blt.ts"
        raise ValueError(f"Unsupported mobileclip variant: {text_model_variant}")
    raise ValueError(f"Unsupported text model variant: {text_model_variant}")


def build_visual_config(
    checkpoint: dict,
    model: torch.nn.Module,
    *,
    visual_dtype: str,
    text_encoder_dir_name: str,
    text_encoder_type: str,
    text_encoder_asset: str,
) -> dict[str, object]:
    train_args = checkpoint.get("train_args") or {}
    yaml = getattr(model, "yaml", {}) or {}
    yaml_file = str(yaml.get("yaml_file") or "yoloe-26x-seg.yaml")
    variant = Path(yaml_file).stem
    segmentation = isinstance(model, UltralyticsYOLOESegModel) or "seg" in variant
    task = "instance-segmentation" if segmentation else "object-detection"
    image_size = int(train_args.get("imgsz") or 640)
    stride = [int(value) for value in getattr(model, "stride", torch.tensor([8, 16, 32])).tolist()]
    return {
        "model_type": "yoloe",
        "architectures": ["YOLOEModel"],
        "variant": variant,
        "task": task,
        "image_size": image_size,
        "num_channels": int(yaml.get("channels", 3)),
        "segmentation": segmentation,
        "fused": bool(model.is_fused()),
        "open_vocab": True,
        "default_classes": [],
        "num_labels": 0,
        "id2label": {},
        "label2id": {},
        "score_threshold": 0.25,
        "iou_threshold": float(train_args.get("iou") or 0.7),
        "stride": stride,
        "model_input_name": "pixel_values",
        "dtype": visual_dtype,
        "text_encoder_type": text_encoder_type,
        "text_encoder_asset": text_encoder_asset,
        "text_encoder_path": f"../{text_encoder_dir_name}",
        "text_embedding_dim": 512,
    }


def build_text_encoder_config(
    *,
    variant: str,
    text_encoder_type: str,
    text_encoder_asset: str,
) -> dict[str, object]:
    return {
        "model_type": "yoloe",
        "architectures": ["MobileCLIPTextEncoder"],
        "variant": variant,
        "task": "text-encoding",
        "dtype": "float32",
        "text_encoder_type": text_encoder_type,
        "text_encoder_asset": text_encoder_asset,
        "text_embedding_dim": 512,
    }


def main() -> None:
    args = parse_args()
    source_model = Path(args.source_model).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = output_root / str(args.model_name)
    text_encoder_dir = output_root / str(args.text_encoder_name)

    checkpoint, model = load_source_model(source_model)
    visual_dtype = infer_visual_dtype(model)
    text_model_variant = str(getattr(model, "text_model", "mobileclip2:b"))
    text_encoder_type, text_encoder_asset = resolve_text_encoder_asset(text_model_variant)
    text_encoder_weight_path = Path(attempt_download_asset(text_encoder_asset))

    model_dir.mkdir(parents=True, exist_ok=True)
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    visual_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(visual_state, str(model_dir / "model.safetensors"))

    asset_bytes = text_encoder_weight_path.read_bytes()
    asset_array = np.frombuffer(asset_bytes, dtype=np.uint8).copy()
    text_state = {
        f"{ASSET_PREFIX}{text_encoder_asset}": torch.from_numpy(asset_array),
    }
    save_file(text_state, str(text_encoder_dir / "model.safetensors"))

    visual_config = build_visual_config(
        checkpoint,
        model,
        visual_dtype=visual_dtype,
        text_encoder_dir_name=text_encoder_dir.name,
        text_encoder_type=text_encoder_type,
        text_encoder_asset=text_encoder_asset,
    )
    text_config = build_text_encoder_config(
        variant=str(visual_config["variant"]),
        text_encoder_type=text_encoder_type,
        text_encoder_asset=text_encoder_asset,
    )

    (model_dir / "config.json").write_text(json.dumps(visual_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (text_encoder_dir / "config.json").write_text(json.dumps(text_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"source_model: {source_model}")
    print(f"visual_dir: {model_dir}")
    print(f"text_encoder_dir: {text_encoder_dir}")
    print(f"visual_tensors: {len(visual_state)}")
    print(f"text_encoder_asset: {text_encoder_asset} ({len(asset_bytes)} bytes)")


if __name__ == "__main__":
    main()
