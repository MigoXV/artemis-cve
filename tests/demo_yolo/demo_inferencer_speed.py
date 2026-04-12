from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from artemis_cve.inferencers.yolo import YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_VIDEO_PATH = Path("data-bin/1082895552-1-208.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark artemis-cve box inferencer on a local video."
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
    parser.add_argument(
        "--compare-cuda-graph",
        action="store_true",
        help="Run baseline and CUDA Graph modes back-to-back and print a comparison summary.",
    )
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
        default=0,
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


def percentile_ms(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)) * 1000.0, 3)


def resolve_device_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )
    return device_index


def run_benchmark(
    *,
    frames: list[np.ndarray],
    class_names: list[str],
    args: argparse.Namespace,
    use_cuda_graph: bool,
    device_index: int | None,
    warmup_frames: int,
) -> dict[str, float | int | str | bool | None]:
    if device_index is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)

    start_load = time.perf_counter()
    inferencer = YoloBoxInferencer(
        model_dir=args.model_dir,
        textencoder_model_dir=args.textencoder_model_dir,
        class_names=class_names,
        device=args.device,
        dtype=args.dtype,
        use_cuda_graph=use_cuda_graph,
    )
    synchronize_device(args.device)
    load_time_s = time.perf_counter() - start_load

    warmup_times: list[float] = []
    measured_times: list[float] = []
    detection_counts: list[int] = []

    progress_desc = "Inferencing[cg]" if inferencer.use_cuda_graph else "Inferencing[base]"
    with tqdm(frames, desc=progress_desc, unit="frame") as progress:
        for index, frame in enumerate(progress):
            start = time.perf_counter()
            detections = inferencer.infer(
                bgr=frame,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
            )
            synchronize_device(args.device)
            elapsed = time.perf_counter() - start
            current_fps = (1.0 / elapsed) if elapsed > 0.0 else float("inf")
            progress.set_postfix_str(f"fps={current_fps:.2f}")

            detection_counts.append(len(detections))
            if index < warmup_frames:
                warmup_times.append(elapsed)
            else:
                measured_times.append(elapsed)

    total_all_s = float(sum(warmup_times) + sum(measured_times))
    total_measured_s = float(sum(measured_times))
    measured_frames = len(measured_times)

    return {
        "device": args.device,
        "dtype": args.dtype,
        "cuda_graph": inferencer.use_cuda_graph,
        "torch_device_name": torch.cuda.get_device_name(device_index)
        if device_index is not None
        else "cpu",
        "load_time_s": round(load_time_s, 4),
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "all_frames": len(frames),
        "steady_infer_fps": round(measured_frames / total_measured_s, 3)
        if total_measured_s > 0.0
        else None,
        "all_frames_fps_after_load": round(len(frames) / total_all_s, 3) if total_all_s > 0.0 else None,
        "avg_infer_ms": round((total_measured_s / measured_frames) * 1000.0, 3)
        if measured_frames > 0
        else None,
        "p50_infer_ms": percentile_ms(measured_times, 50),
        "p95_infer_ms": percentile_ms(measured_times, 95),
        "peak_mem_mb": round(float(torch.cuda.max_memory_allocated(device_index) / (1024**2)), 1)
        if device_index is not None
        else None,
        "avg_detections_per_frame": round(float(sum(detection_counts) / len(detection_counts)), 3),
    }


def build_comparison(
    baseline: dict[str, float | int | str | bool | None],
    cuda_graph: dict[str, float | int | str | bool | None],
) -> dict[str, float | int | None]:
    baseline_avg_ms = baseline.get("avg_infer_ms")
    graph_avg_ms = cuda_graph.get("avg_infer_ms")
    baseline_fps = baseline.get("steady_infer_fps")
    graph_fps = cuda_graph.get("steady_infer_fps")

    avg_ms_speedup = None
    if isinstance(baseline_avg_ms, (int, float)) and isinstance(graph_avg_ms, (int, float)) and graph_avg_ms > 0:
        avg_ms_speedup = round(float(baseline_avg_ms) / float(graph_avg_ms), 3)

    fps_speedup = None
    if isinstance(baseline_fps, (int, float)) and isinstance(graph_fps, (int, float)) and baseline_fps > 0:
        fps_speedup = round(float(graph_fps) / float(baseline_fps), 3)

    avg_ms_delta = None
    if isinstance(baseline_avg_ms, (int, float)) and isinstance(graph_avg_ms, (int, float)):
        avg_ms_delta = round(float(graph_avg_ms) - float(baseline_avg_ms), 3)

    return {
        "baseline_avg_infer_ms": baseline_avg_ms if isinstance(baseline_avg_ms, (int, float)) else None,
        "cuda_graph_avg_infer_ms": graph_avg_ms if isinstance(graph_avg_ms, (int, float)) else None,
        "avg_infer_ms_delta": avg_ms_delta,
        "avg_infer_ms_speedup": avg_ms_speedup,
        "baseline_steady_infer_fps": baseline_fps if isinstance(baseline_fps, (int, float)) else None,
        "cuda_graph_steady_infer_fps": graph_fps if isinstance(graph_fps, (int, float)) else None,
        "steady_infer_fps_speedup": fps_speedup,
    }


def main() -> None:
    args = parse_args()

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    if not class_names:
        raise ValueError("At least one class name is required.")

    video_path = Path(args.video)
    frames, video_meta = decode_video_frames(video_path, max_frames=max(0, int(args.max_frames)))
    warmup_frames = min(max(0, int(args.warmup_frames)), len(frames))
    device_index = resolve_device_index(args.device)

    print(f"video: {video_path}")
    print(f"video_meta: {json.dumps(video_meta, ensure_ascii=False)}")
    print(f"decoded_frames: {len(frames)}")
    print(f"class_names: {class_names}")
    print(f"device: {args.device}")
    print(f"dtype: {args.dtype}")
    print(f"cuda_graph: {args.cuda_graph}")
    print(f"compare_cuda_graph: {args.compare_cuda_graph}")
    print(f"warmup_frames: {warmup_frames}")
    if device_index is not None:
        print(f"torch_device_name: {torch.cuda.get_device_name(device_index)}")

    if args.compare_cuda_graph:
        baseline_result = run_benchmark(
            frames=frames,
            class_names=class_names,
            args=args,
            use_cuda_graph=False,
            device_index=device_index,
            warmup_frames=warmup_frames,
        )
        print(f"RESULT_BASELINE {json.dumps(baseline_result, ensure_ascii=False)}")

        cuda_graph_result = run_benchmark(
            frames=frames,
            class_names=class_names,
            args=args,
            use_cuda_graph=True,
            device_index=device_index,
            warmup_frames=warmup_frames,
        )
        print(f"RESULT_CUDA_GRAPH {json.dumps(cuda_graph_result, ensure_ascii=False)}")

        comparison = build_comparison(baseline=baseline_result, cuda_graph=cuda_graph_result)
        print(f"RESULT_COMPARE {json.dumps(comparison, ensure_ascii=False)}")
        return

    explicit_cuda_graph = bool(device_index is not None) if args.cuda_graph is None else bool(args.cuda_graph)
    result = run_benchmark(
        frames=frames,
        class_names=class_names,
        args=args,
        use_cuda_graph=explicit_cuda_graph,
        device_index=device_index,
        warmup_frames=warmup_frames,
    )
    print(f"RESULT {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
