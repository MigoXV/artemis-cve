from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from transformers import AutoModel
from ultralytics.nn.tasks import YOLOEModel as UltralyticsYOLOEModel
from ultralytics.nn.tasks import YOLOESegModel as UltralyticsYOLOESegModel
from ultralytics.utils import nms
from ultralytics.utils.ops import scale_boxes

import artemis_cve.models
from artemis_cve.inferencers.yolo import BoxDetection, YoloBoxInferencer
from artemis_cve.models.yolo26e import YOLOEConfig
from artemis_cve.models.mobileclip2 import YOLOETextEncoder
from artemis_cve.models.yolo26e import YOLOERawOutput, forward_yoloe_task_model_raw

DEFAULT_IMAGE_DIR = Path("data-bin/images")
DEFAULT_CLASS_NAMES_FILE = Path("model-bin/names.txt")
DEFAULT_HF_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_HF_TEXT_ENCODER_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_OPENVISION_MODEL = Path("model-bin/openvision/yoloe26-x-seg/model.pt")
DEFAULT_OUTPUT_ROOT = Path("data-bin/align")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(slots=True)
class ComparableDetection:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align pure HF YOLOE model against Ultralytics YOLOE baselines.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--class-names-file", default=str(DEFAULT_CLASS_NAMES_FILE))
    parser.add_argument("--hf-model-dir", default=str(DEFAULT_HF_MODEL_DIR))
    parser.add_argument("--hf-text-encoder-dir", default=str(DEFAULT_HF_TEXT_ENCODER_DIR))
    parser.add_argument("--openvision-model", default=str(DEFAULT_OPENVISION_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def parse_class_names(class_names_file: str) -> list[str]:
    path = Path(class_names_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"class names file does not exist: {path}")

    class_names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"class names file is empty: {path}")
    return class_names


def collect_images(image_dir: str) -> list[Path]:
    directory = Path(image_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no supported images found in: {directory}")
    return image_paths


def build_report_path(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"align01-{timestamp}.json"


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
    max_abs_diff = 0.0
    mismatched: list[str] = []

    for key in sorted(lhs_keys & rhs_keys):
        left = lhs[key].detach().float()
        right = rhs[key].detach().float()
        if tuple(left.shape) != tuple(right.shape):
            mismatched.append(key)
            continue
        diff = float((left - right).abs().max().item()) if left.numel() else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
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


def sort_detections(detections: ComparableDetection) -> ComparableDetection:
    if detections.scores.numel() == 0:
        return detections
    order = sorted(
        range(int(detections.scores.shape[0])),
        key=lambda idx: (
            -float(detections.scores[idx].item()),
            int(detections.labels[idx].item()),
            float(detections.boxes[idx][0].item()),
            float(detections.boxes[idx][1].item()),
            float(detections.boxes[idx][2].item()),
            float(detections.boxes[idx][3].item()),
        ),
    )
    index_tensor = torch.tensor(order, dtype=torch.long)
    return ComparableDetection(
        boxes=detections.boxes[index_tensor],
        scores=detections.scores[index_tensor],
        labels=detections.labels[index_tensor],
    )


def detections_from_ultralytics_results(results) -> ComparableDetection:
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return ComparableDetection(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )
    return sort_detections(
        ComparableDetection(
            boxes=boxes.xyxy.detach().cpu().to(dtype=torch.float32),
            scores=boxes.conf.detach().cpu().to(dtype=torch.float32),
            labels=boxes.cls.detach().cpu().to(dtype=torch.long),
        )
    )


def detections_from_box_detections(detections: list[BoxDetection]) -> ComparableDetection:
    if not detections:
        return ComparableDetection(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )
    return sort_detections(
        ComparableDetection(
            boxes=torch.tensor([item.pixel_xyxy for item in detections], dtype=torch.float32),
            scores=torch.tensor([item.score for item in detections], dtype=torch.float32),
            labels=torch.tensor([item.class_id for item in detections], dtype=torch.long),
        )
    )


def detections_from_processed_prediction(
    prediction: torch.Tensor,
    *,
    original_shape: tuple[int, int],
    processed_shape: tuple[int, int],
) -> ComparableDetection:
    if prediction.numel() == 0:
        return ComparableDetection(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )
    scaled_boxes = scale_boxes(processed_shape, prediction[:, :4].clone(), original_shape)
    return sort_detections(
        ComparableDetection(
            boxes=scaled_boxes.detach().cpu().to(dtype=torch.float32),
            scores=prediction[:, 4].detach().cpu().to(dtype=torch.float32),
            labels=prediction[:, 5].detach().cpu().to(dtype=torch.long),
        )
    )


def compare_detections(
    lhs: ComparableDetection,
    rhs: ComparableDetection,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    boxes_match, boxes_diff = tensor_equal(lhs.boxes, rhs.boxes, atol=atol, rtol=rtol)
    scores_match, scores_diff = tensor_equal(lhs.scores, rhs.scores, atol=atol, rtol=rtol)
    labels_match = torch.equal(lhs.labels, rhs.labels)
    passed = boxes_match and scores_match and labels_match
    reasons: list[str] = []
    if not boxes_match:
        reasons.append("boxes")
    if not scores_match:
        reasons.append("scores")
    if not labels_match:
        reasons.append("labels")
    return {
        "passed": passed,
        "count_left": int(lhs.scores.shape[0]),
        "count_right": int(rhs.scores.shape[0]),
        "box_max_abs_diff": boxes_diff,
        "score_max_abs_diff": scores_diff,
        "reasons": reasons,
    }


def compare_raw_outputs(
    hf_output: YOLOERawOutput,
    baseline_output: YOLOERawOutput,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    comparisons = {
        "box_logits": tensor_equal(hf_output.box_logits, baseline_output.box_logits, atol=atol, rtol=rtol),
        "class_logits": tensor_equal(hf_output.class_logits, baseline_output.class_logits, atol=atol, rtol=rtol),
        "mask_coefficients": tensor_equal(
            hf_output.mask_coefficients,
            baseline_output.mask_coefficients,
            atol=atol,
            rtol=rtol,
        ),
        "prototypes": tensor_equal(hf_output.prototypes, baseline_output.prototypes, atol=atol, rtol=rtol),
    }
    reasons = [
        name
        for name, (matched, _) in comparisons.items()
        if not matched
    ]
    if hf_output.feature_shapes != baseline_output.feature_shapes:
        reasons.append("feature_shapes")
    if hf_output.strides != baseline_output.strides:
        reasons.append("strides")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "box_logits_max_abs_diff": comparisons["box_logits"][1],
        "class_logits_max_abs_diff": comparisons["class_logits"][1],
        "mask_coefficients_max_abs_diff": comparisons["mask_coefficients"][1],
        "prototypes_max_abs_diff": comparisons["prototypes"][1],
        "feature_shapes_left": hf_output.feature_shapes,
        "feature_shapes_right": baseline_output.feature_shapes,
        "strides_left": hf_output.strides,
        "strides_right": baseline_output.strides,
        "hf_box_logits_preview": preview_tensor(hf_output.box_logits),
        "baseline_box_logits_preview": preview_tensor(baseline_output.box_logits),
    }


def main() -> None:
    args = parse_args()
    class_names = parse_class_names(args.class_names_file)
    image_paths = collect_images(args.image_dir)
    report_path = build_report_path(args.output_root)
    device = torch.device(args.device)

    hf_model = AutoModel.from_pretrained(args.hf_model_dir)
    hf_model.to(device=device, dtype=torch.float32)
    hf_model.eval()

    hf_config = YOLOEConfig.from_pretrained(args.hf_model_dir)
    openvision_checkpoint = torch.load(args.openvision_model, map_location="cpu", weights_only=False)
    baseline_model = openvision_checkpoint["model"] if isinstance(openvision_checkpoint, dict) else openvision_checkpoint
    if not isinstance(baseline_model, (UltralyticsYOLOESegModel, UltralyticsYOLOEModel)):
        raise TypeError(f"Unsupported openvision model type: {type(baseline_model)!r}")
    baseline_model = baseline_model.float().to(device=device, dtype=torch.float32)
    baseline_model.eval()

    state_dict_report = compare_state_dicts(
        {
            key: value.detach().cpu()
            for key, value in hf_model.state_dict().items()
        },
        {
            key: value.detach().cpu()
            for key, value in baseline_model.state_dict().items()
        },
        atol=args.atol,
        rtol=args.rtol,
    )

    text_encoder = YOLOETextEncoder.from_pretrained(
        args.hf_text_encoder_dir,
        device=device,
    )
    base_text_embeddings = text_encoder.encode(class_names, dtype=torch.float32)
    hf_prompt_embeddings = hf_model.project_text_embeddings(base_text_embeddings)

    baseline_prompt_embeddings = baseline_model.model[-1].get_tpe(
        base_text_embeddings.to(device=device, dtype=torch.float32).clone()
    )

    text_align_match, text_align_diff = tensor_equal(
        base_text_embeddings,
        baseline_model.get_text_pe(class_names, without_reprta=True).to(device=device, dtype=torch.float32),
        atol=args.atol,
        rtol=args.rtol,
    )
    prompt_align_match, prompt_align_diff = tensor_equal(
        hf_prompt_embeddings,
        baseline_prompt_embeddings,
        atol=args.atol,
        rtol=args.rtol,
    )

    inferencer = YoloBoxInferencer(
        model_dir=args.hf_model_dir,
        textencoder_model_dir=args.hf_text_encoder_dir,
        class_names=class_names,
        device=device,
        dtype="fp32",
        use_cuda_graph=False,
    )

    image_reports: list[dict[str, Any]] = []
    all_passed = state_dict_report["passed"] and text_align_match and prompt_align_match

    print(f"image_dir: {Path(args.image_dir).expanduser()}")
    print(f"class_names: {class_names}")
    print(f"hf_model_dir: {Path(args.hf_model_dir).expanduser()}")
    print(f"openvision_model: {Path(args.openvision_model).expanduser()}")
    print(f"device: {device}")
    print(
        "state_dict_align: "
        f"passed={state_dict_report['passed']} max_abs_diff={state_dict_report['max_abs_diff']}"
    )
    print(f"text_embedding_align: passed={text_align_match} max_abs_diff={text_align_diff}")
    print(f"prompt_embedding_align: passed={prompt_align_match} max_abs_diff={prompt_align_diff}")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")

        pixel_values, original_shape, processed_shape = inferencer._preprocess(image_bgr)
        hf_raw = hf_model(
            pixel_values,
            text_embeddings=base_text_embeddings,
        )
        baseline_raw = forward_yoloe_task_model_raw(
            baseline_model,
            pixel_values,
            text_embeddings=base_text_embeddings,
        )
        raw_report = compare_raw_outputs(
            hf_raw,
            baseline_raw,
            atol=args.atol,
            rtol=args.rtol,
        )

        hf_box_detections = inferencer.infer(
            image_bgr,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )
        hf_detections = detections_from_box_detections(hf_box_detections)
        with torch.no_grad():
            baseline_processed, _ = baseline_model(pixel_values, tpe=base_text_embeddings)
        if isinstance(baseline_processed, tuple):
            baseline_processed = baseline_processed[0]
        filtered = nms.non_max_suppression(
            baseline_processed,
            conf_thres=args.score_threshold,
            iou_thres=args.iou_threshold,
            agnostic=True,
            max_det=args.max_detections,
            nc=len(class_names),
            end2end=True,
        )
        baseline_detections = detections_from_processed_prediction(
            filtered[0],
            original_shape=original_shape,
            processed_shape=processed_shape,
        )
        decoded_report = compare_detections(
            hf_detections,
            baseline_detections,
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

        raw_status = "PASS" if raw_report["passed"] else "FAIL"
        decoded_status = "PASS" if decoded_report["passed"] else "FAIL"
        print(f"{image_path.name} raw={raw_status} decoded={decoded_status}")

    report = {
        "passed": all_passed,
        "class_names": class_names,
        "state_dict_alignment": state_dict_report,
        "text_embedding_alignment": {
            "passed": text_align_match,
            "max_abs_diff": text_align_diff,
            "hf_preview": preview_tensor(base_text_embeddings),
        },
        "prompt_embedding_alignment": {
            "passed": prompt_align_match,
            "max_abs_diff": prompt_align_diff,
            "hf_preview": preview_tensor(hf_prompt_embeddings),
            "baseline_preview": preview_tensor(baseline_prompt_embeddings),
        },
        "images": image_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"report: {report_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
