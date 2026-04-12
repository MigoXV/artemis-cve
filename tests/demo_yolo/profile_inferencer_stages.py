from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from artemis_cve.inferencers.yolo import BoxDetection, YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_VIDEO_PATH = Path("data-bin/1082895552-1-208.mp4")
STAGE_ORDER = ("preprocess_cpu", "host_to_device", "forward", "postprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile artemis-cve YOLO inferencer stages on a local video."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO_PATH),
        help="Input video path.",
    )
    parser.add_argument(
        "--class-names",
        default="car,person",
        help="Comma-separated class names for open-vocabulary inference.",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--textencoder-model-dir", default=str(DEFAULT_TEXT_ENCODER_MODEL_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--cuda-graph", dest="cuda_graph", action="store_true")
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    parser.set_defaults(cuda_graph=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Frames to exclude from steady-state timing statistics.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Optional frame cap after decoding. Use 0 to process the full video.",
    )
    return parser.parse_args()


def decode_video_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], dict[str, float | int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    meta = {
        "video_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count_meta": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, meta


def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_call(device: str, fn) -> tuple[Any, float]:
    synchronize_device(device)
    start = time.perf_counter()
    result = fn()
    synchronize_device(device)
    return result, time.perf_counter() - start


def percentile_ms(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)) * 1000.0, 3)


def summarize_stage(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "sum_ms": None,
        }

    total_s = float(sum(values))
    count = len(values)
    return {
        "avg_ms": round((total_s / count) * 1000.0, 3),
        "p50_ms": percentile_ms(values, 50),
        "p95_ms": percentile_ms(values, 95),
        "sum_ms": round(total_s * 1000.0, 3),
    }


def preprocess_cpu(
    inferencer: YoloBoxInferencer,
    bgr: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(
            "Expected a BGR image with shape (H, W, 3), "
            f"received {tuple(bgr.shape)}"
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    processed = inferencer._letterbox(image=rgb)
    return processed, tuple(int(v) for v in rgb.shape[:2]), tuple(int(v) for v in processed.shape[:2])


def host_to_device(
    inferencer: YoloBoxInferencer,
    processed: np.ndarray,
) -> torch.Tensor:
    return (
        torch.from_numpy(processed)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=inferencer.device, dtype=inferencer.dtype)
        / 255.0
    )


def main() -> None:
    args = parse_args()

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    if not class_names:
        raise ValueError("At least one class name is required.")

    video_path = Path(args.video)
    frames, video_meta = decode_video_frames(video_path, max_frames=max(0, int(args.max_frames)))
    warmup_frames = min(max(0, int(args.warmup_frames)), len(frames))

    print(f"video: {video_path}")
    print(f"video_meta: {json.dumps(video_meta, ensure_ascii=False)}")
    print(f"decoded_frames: {len(frames)}")
    print(f"class_names: {class_names}")
    print(f"device: {args.device}")
    print(f"dtype: {args.dtype}")
    print(f"cuda_graph: {args.cuda_graph}")
    print(f"warmup_frames: {warmup_frames}")

    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {args.device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        print(f"torch_device_name: {torch.cuda.get_device_name(device_index)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
    else:
        device_index = None

    start_load = time.perf_counter()
    inferencer = YoloBoxInferencer(
        model_dir=args.model_dir,
        textencoder_model_dir=args.textencoder_model_dir,
        class_names=class_names,
        device=args.device,
        dtype=args.dtype,
        use_cuda_graph=args.cuda_graph,
    )
    synchronize_device(args.device)
    load_time_s = time.perf_counter() - start_load
    print(f"load_time_s: {load_time_s:.4f}")

    stage_values: dict[str, list[float]] = {name: [] for name in STAGE_ORDER}
    total_values: list[float] = []
    detection_counts: list[int] = []

    with tqdm(frames, desc="Profiling", unit="frame") as progress:
        for index, frame in enumerate(progress):
            (processed, original_shape, processed_shape), preprocess_cpu_s = timed_call(
                args.device,
                lambda frame=frame: preprocess_cpu(inferencer, frame),
            )
            tensor, host_to_device_s = timed_call(
                args.device,
                lambda processed=processed: host_to_device(inferencer, processed),
            )

            outputs, forward_s = timed_call(
                args.device,
                lambda: inferencer._forward_raw(
                    pixel_values=tensor,
                    max_detections=args.max_detections,
                ),
            )
            detections, postprocess_s = timed_call(
                args.device,
                lambda: inferencer._convert_outputs(
                    outputs=outputs,
                    original_shape=original_shape,
                    processed_shape=processed_shape,
                    score_threshold=float(args.score_threshold),
                    max_detections=args.max_detections,
                ),
            )

            total_s = preprocess_cpu_s + host_to_device_s + forward_s + postprocess_s
            if index >= warmup_frames:
                stage_values["preprocess_cpu"].append(preprocess_cpu_s)
                stage_values["host_to_device"].append(host_to_device_s)
                stage_values["forward"].append(forward_s)
                stage_values["postprocess"].append(postprocess_s)
                total_values.append(total_s)
                detection_counts.append(len(detections))

            if total_s > 0.0:
                progress.set_postfix_str(
                    " ".join(
                        [
                            f"total={total_s * 1000.0:.1f}ms",
                            f"pre={preprocess_cpu_s * 1000.0:.1f}",
                            f"h2d={host_to_device_s * 1000.0:.1f}",
                            f"fwd={forward_s * 1000.0:.1f}",
                            f"post={postprocess_s * 1000.0:.1f}",
                        ]
                    )
                )

    stage_summary = {name: summarize_stage(values) for name, values in stage_values.items()}
    total_measured_s = float(sum(total_values))
    for name in STAGE_ORDER:
        share = None
        if total_measured_s > 0.0:
            share = round(float(sum(stage_values[name])) / total_measured_s * 100.0, 2)
        stage_summary[name]["share_percent"] = share

    bottleneck = max(
        STAGE_ORDER,
        key=lambda stage_name: float(sum(stage_values[stage_name])) if stage_values[stage_name] else -1.0,
    )

    result = {
        "device": args.device,
        "dtype": args.dtype,
        "cuda_graph": inferencer.use_cuda_graph,
        "torch_device_name": torch.cuda.get_device_name(device_index)
        if device_index is not None
        else "cpu",
        "load_time_s": round(load_time_s, 4),
        "warmup_frames": warmup_frames,
        "measured_frames": len(total_values),
        "all_frames": len(frames),
        "steady_total_fps": round(len(total_values) / total_measured_s, 3)
        if total_measured_s > 0.0
        else None,
        "avg_total_ms": round((total_measured_s / len(total_values)) * 1000.0, 3)
        if total_values
        else None,
        "p50_total_ms": percentile_ms(total_values, 50),
        "p95_total_ms": percentile_ms(total_values, 95),
        "peak_mem_mb": round(float(torch.cuda.max_memory_allocated(device_index) / (1024**2)), 1)
        if device_index is not None
        else None,
        "avg_detections_per_frame": round(float(sum(detection_counts) / len(detection_counts)), 3)
        if detection_counts
        else None,
        "bottleneck_stage": bottleneck,
        "stage_breakdown": stage_summary,
    }
    print(f"RESULT {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
