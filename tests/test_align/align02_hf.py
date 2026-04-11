from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from transformers import AutoModel
from ultralytics.data.augment import LetterBox

import artemis_cve.models
from artemis_cve.inferencers.yolo.runtime import BaseYoloRuntime, RawYoloOutput
from artemis_cve.models.mobileclip2 import YOLOETextEncoder
from artemis_cve.models.yolo26e import YOLOEConfig, YOLOERawOutput

DEFAULT_IMAGE_DIR = Path("data-bin/images")
DEFAULT_CLASS_NAMES_FILE = Path("model-bin/names.txt")
DEFAULT_LOCAL_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_LOCAL_TEXT_ENCODER_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_HF_MODEL_ID = "MigoXV/yoloe26-x-seg"
DEFAULT_HF_TEXT_ENCODER_ID = "MigoXV/mobileclip2-b"
DEFAULT_OUTPUT_ROOT = Path("data-bin/align")
DEFAULT_HF_TOKEN_ENV = "HF_TOKEN"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align local MigoXV checkpoints against HF from_pretrained loading.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--class-names-file", default=str(DEFAULT_CLASS_NAMES_FILE))
    parser.add_argument("--local-model-dir", default=str(DEFAULT_LOCAL_MODEL_DIR))
    parser.add_argument("--local-text-encoder-dir", default=str(DEFAULT_LOCAL_TEXT_ENCODER_DIR))
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-text-encoder-id", default=DEFAULT_HF_TEXT_ENCODER_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default=DEFAULT_HF_TOKEN_ENV)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def parse_class_names(class_names_file: str) -> list[str]:
    path = Path(class_names_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"class names file does not exist: {path}")
    class_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"class names file is empty: {path}")
    return class_names


def collect_images(image_dir: str) -> list[Path]:
    directory = Path(image_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    image_paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no supported images found in: {directory}")
    return image_paths


def build_report_path(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"align02-hf-{timestamp}.json"


def tensor_equal(lhs: torch.Tensor | None, rhs: torch.Tensor | None, *, atol: float, rtol: float) -> tuple[bool, float | None]:
    if lhs is None and rhs is None:
        return True, None
    if lhs is None or rhs is None:
        return False, None
    if tuple(lhs.shape) != tuple(rhs.shape):
        return False, None
    diff = float((lhs.detach().float() - rhs.detach().float()).abs().max().item()) if lhs.numel() else 0.0
    return torch.allclose(lhs.detach().float(), rhs.detach().float(), atol=atol, rtol=rtol), diff


def preview_tensor(tensor: torch.Tensor | None, *, limit: int = 6) -> list[float] | None:
    if tensor is None:
        return None
    flat = tensor.detach().float().reshape(-1)
    return [round(float(value), 6) for value in flat[:limit].tolist()]


def compare_state_dicts(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    lhs_keys = set(lhs)
    rhs_keys = set(rhs)
    missing = sorted(rhs_keys - lhs_keys)
    unexpected = sorted(lhs_keys - rhs_keys)
    mismatched: list[str] = []
    max_abs_diff = 0.0
    for key in sorted(lhs_keys & rhs_keys):
        matched, diff = tensor_equal(lhs[key], rhs[key], atol=atol, rtol=rtol)
        if diff is not None:
            max_abs_diff = max(max_abs_diff, diff)
        if not matched:
            mismatched.append(key)
    return {
        "passed": not missing and not unexpected and not mismatched,
        "missing_keys": missing[:20],
        "unexpected_keys": unexpected[:20],
        "mismatched_keys": mismatched[:20],
        "max_abs_diff": max_abs_diff,
        "num_keys_left": len(lhs),
        "num_keys_right": len(rhs),
    }


def compare_raw_outputs(
    lhs: YOLOERawOutput,
    rhs: YOLOERawOutput,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    comparisons = {
        "box_logits": tensor_equal(lhs.box_logits, rhs.box_logits, atol=atol, rtol=rtol),
        "class_logits": tensor_equal(lhs.class_logits, rhs.class_logits, atol=atol, rtol=rtol),
        "mask_coefficients": tensor_equal(lhs.mask_coefficients, rhs.mask_coefficients, atol=atol, rtol=rtol),
        "prototypes": tensor_equal(lhs.prototypes, rhs.prototypes, atol=atol, rtol=rtol),
    }
    reasons = [name for name, (matched, _) in comparisons.items() if not matched]
    if lhs.feature_shapes != rhs.feature_shapes:
        reasons.append("feature_shapes")
    if lhs.strides != rhs.strides:
        reasons.append("strides")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "box_logits_max_abs_diff": comparisons["box_logits"][1],
        "class_logits_max_abs_diff": comparisons["class_logits"][1],
        "mask_coefficients_max_abs_diff": comparisons["mask_coefficients"][1],
        "prototypes_max_abs_diff": comparisons["prototypes"][1],
        "feature_shapes_left": lhs.feature_shapes,
        "feature_shapes_right": rhs.feature_shapes,
        "strides_left": lhs.strides,
        "strides_right": rhs.strides,
        "left_box_logits_preview": preview_tensor(lhs.box_logits),
        "right_box_logits_preview": preview_tensor(rhs.box_logits),
    }


def compare_decoded_outputs(
    lhs: RawYoloOutput,
    rhs: RawYoloOutput,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    lhs_boxes = lhs.boxes[0] if lhs.boxes else torch.empty((0, 4), dtype=torch.float32)
    rhs_boxes = rhs.boxes[0] if rhs.boxes else torch.empty((0, 4), dtype=torch.float32)
    lhs_scores = lhs.scores[0] if lhs.scores else torch.empty((0,), dtype=torch.float32)
    rhs_scores = rhs.scores[0] if rhs.scores else torch.empty((0,), dtype=torch.float32)
    lhs_labels = lhs.labels[0] if lhs.labels else torch.empty((0,), dtype=torch.long)
    rhs_labels = rhs.labels[0] if rhs.labels else torch.empty((0,), dtype=torch.long)
    lhs_masks = lhs.masks[0] if lhs.masks else None
    rhs_masks = rhs.masks[0] if rhs.masks else None

    box_match, box_diff = tensor_equal(lhs_boxes, rhs_boxes, atol=atol, rtol=rtol)
    score_match, score_diff = tensor_equal(lhs_scores, rhs_scores, atol=atol, rtol=rtol)
    mask_match, mask_diff = tensor_equal(lhs_masks, rhs_masks, atol=atol, rtol=rtol)
    label_match = torch.equal(lhs_labels.detach().cpu(), rhs_labels.detach().cpu())
    reasons: list[str] = []
    if not box_match:
        reasons.append("boxes")
    if not score_match:
        reasons.append("scores")
    if not label_match:
        reasons.append("labels")
    if not mask_match:
        reasons.append("masks")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "count_left": int(lhs_scores.shape[0]),
        "count_right": int(rhs_scores.shape[0]),
        "box_max_abs_diff": box_diff,
        "score_max_abs_diff": score_diff,
        "mask_max_abs_diff": mask_diff,
    }


def preprocess_image(
    image_bgr: Any,
    *,
    image_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterbox = LetterBox(
        new_shape=(image_size, image_size),
        auto=False,
        scale_fill=False,
        scaleup=True,
        stride=stride,
    )
    processed = letterbox(image=rgb)
    return (
        torch.from_numpy(processed)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    if args.hf_token:
        return args.hf_token
    env_name = str(args.hf_token_env).strip()
    if not env_name:
        return None
    return os.environ.get(env_name)


def main() -> None:
    args = parse_args()
    class_names = parse_class_names(args.class_names_file)
    image_paths = collect_images(args.image_dir)
    report_path = build_report_path(args.output_root)
    device = torch.device(args.device)
    token = resolve_hf_token(args)

    local_model = AutoModel.from_pretrained(args.local_model_dir)
    remote_model = AutoModel.from_pretrained(args.hf_model_id, token=token)
    local_model.to(device=device, dtype=torch.float32).eval()
    remote_model.to(device=device, dtype=torch.float32).eval()

    local_text_encoder = YOLOETextEncoder.from_pretrained(
        args.local_text_encoder_dir,
        device=device,
        token=token,
    )
    remote_text_encoder = YOLOETextEncoder.from_pretrained(
        args.hf_text_encoder_id,
        device=device,
        token=token,
    )

    local_config = YOLOEConfig.from_pretrained(args.local_model_dir)
    remote_config = YOLOEConfig.from_pretrained(args.hf_model_id, token=token)

    state_dict_report = compare_state_dicts(
        {key: value.detach().cpu() for key, value in local_model.state_dict().items()},
        {key: value.detach().cpu() for key, value in remote_model.state_dict().items()},
        atol=args.atol,
        rtol=args.rtol,
    )

    local_text_embeddings = local_text_encoder.encode(class_names, dtype=torch.float32)
    remote_text_embeddings = remote_text_encoder.encode(class_names, dtype=torch.float32)
    text_align_match, text_align_diff = tensor_equal(
        local_text_embeddings,
        remote_text_embeddings,
        atol=args.atol,
        rtol=args.rtol,
    )

    local_prompt_embeddings = local_model.project_text_embeddings(local_text_embeddings)
    remote_prompt_embeddings = remote_model.project_text_embeddings(remote_text_embeddings)
    prompt_align_match, prompt_align_diff = tensor_equal(
        local_prompt_embeddings,
        remote_prompt_embeddings,
        atol=args.atol,
        rtol=args.rtol,
    )

    local_runtime = BaseYoloRuntime(local_model)
    remote_runtime = BaseYoloRuntime(remote_model)
    stride = int(max(int(value) for value in getattr(local_config, "stride", [32])))
    image_size = int(getattr(local_config, "image_size", 640))

    all_passed = state_dict_report["passed"] and text_align_match and prompt_align_match
    image_reports: list[dict[str, Any]] = []

    print(f"local_model_dir: {Path(args.local_model_dir).expanduser()}")
    print(f"local_text_encoder_dir: {Path(args.local_text_encoder_dir).expanduser()}")
    print(f"hf_model_id: {args.hf_model_id}")
    print(f"hf_text_encoder_id: {args.hf_text_encoder_id}")
    print(f"device: {device}")
    print(f"state_dict_align: passed={state_dict_report['passed']} max_abs_diff={state_dict_report['max_abs_diff']}")
    print(f"text_embedding_align: passed={text_align_match} max_abs_diff={text_align_diff}")
    print(f"prompt_embedding_align: passed={prompt_align_match} max_abs_diff={prompt_align_diff}")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        pixel_values = preprocess_image(
            image_bgr,
            image_size=image_size,
            stride=stride,
            device=device,
        )

        local_raw = local_model(pixel_values=pixel_values, text_embeddings=local_text_embeddings)
        remote_raw = remote_model(pixel_values=pixel_values, text_embeddings=remote_text_embeddings)
        raw_report = compare_raw_outputs(
            local_raw,
            remote_raw,
            atol=args.atol,
            rtol=args.rtol,
        )

        local_decoded = local_runtime.forward(
            pixel_values,
            text_embeddings=local_text_embeddings,
            max_det=args.max_detections,
        )
        remote_decoded = remote_runtime.forward(
            pixel_values,
            text_embeddings=remote_text_embeddings,
            max_det=args.max_detections,
        )
        decoded_report = compare_decoded_outputs(
            local_decoded,
            remote_decoded,
            atol=args.atol,
            rtol=args.rtol,
        )

        image_report = {
            "image": image_path.name,
            "raw": raw_report,
            "decoded": decoded_report,
        }
        image_reports.append(image_report)
        all_passed = all_passed and raw_report["passed"] and decoded_report["passed"]
        print(
            f"{image_path.name} raw={'PASS' if raw_report['passed'] else 'FAIL'} "
            f"decoded={'PASS' if decoded_report['passed'] else 'FAIL'}"
        )

    report = {
        "passed": all_passed,
        "class_names": class_names,
        "local_model_dir": str(Path(args.local_model_dir).expanduser()),
        "local_text_encoder_dir": str(Path(args.local_text_encoder_dir).expanduser()),
        "hf_model_id": args.hf_model_id,
        "hf_text_encoder_id": args.hf_text_encoder_id,
        "state_dict_alignment": state_dict_report,
        "text_embedding_alignment": {
            "passed": text_align_match,
            "max_abs_diff": text_align_diff,
            "left_preview": preview_tensor(local_text_embeddings),
            "right_preview": preview_tensor(remote_text_embeddings),
        },
        "prompt_embedding_alignment": {
            "passed": prompt_align_match,
            "max_abs_diff": prompt_align_diff,
            "left_preview": preview_tensor(local_prompt_embeddings),
            "right_preview": preview_tensor(remote_prompt_embeddings),
        },
        "config_alignment": {
            "passed": local_config.to_dict() == remote_config.to_dict(),
        },
        "images": image_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {report_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
