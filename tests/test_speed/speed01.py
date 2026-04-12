from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from artemis_cve.inferencers.yolo import YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_CLASS_NAMES_FILE = Path("model-bin/names.txt")
DEFAULT_VIDEO_PATH = Path("data-bin/1082895552-1-208.mp4")
DEFAULT_OUTPUT_ROOT = Path("data-bin/speed")
DEFAULT_DEVICES = ("cuda:0", "cuda:1")
DEFAULT_MODES = ("base", "cuda_graph")
DEFAULT_VIDEO_OUTPUT_ROOT = Path("data-bin/outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inferencer latency on local MigoXV repos across multiple CUDA devices."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--textencoder-model-dir", default=str(DEFAULT_TEXT_ENCODER_MODEL_DIR))
    parser.add_argument("--class-names-file", default=str(DEFAULT_CLASS_NAMES_FILE))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=("base", "cuda_graph"),
        help="Benchmark both standard runtime and CUDA Graph runtime by default.",
    )
    parser.add_argument("--dtype", default="fp16", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap after decoding. Use 0 to process the full video.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video-output-root", default=str(DEFAULT_VIDEO_OUTPUT_ROOT))
    parser.add_argument("--worker-json", default="")
    parser.add_argument("--existing-video-output-dir", default="")
    return parser.parse_args()


def parse_class_names(path_str: str) -> list[str]:
    path = Path(path_str).expanduser()
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


def decode_video_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], dict[str, float | int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")

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
        raise RuntimeError(f"no frames decoded from video: {video_path}")
    return frames, meta


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_ms(values_s: list[float], percentile: float) -> float | None:
    if not values_s:
        return None
    return round(float(np.percentile(values_s, percentile)) * 1000.0, 3)


def build_report_path(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_root).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"speed01-{timestamp}.json"


def build_video_output_dir(output_root: str) -> Path:
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(100):
        candidate = root / (timestamp if suffix == 0 else f"{timestamp}-{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"failed to allocate unique output directory under: {root}")


def draw_boxes(image: np.ndarray, detections: list[object]) -> np.ndarray:
    visualized = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.pixel_xyxy]
        cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            visualized,
            f"{detection.class_name} {detection.score:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return visualized


class H264VideoWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.output_path = output_path
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._process = subprocess.Popen(
            [
                ffmpeg_exe,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError(f"failed to open ffmpeg stdin for: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame, got {frame.dtype}")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self._process.stdin.write(frame.tobytes())

    def release(self) -> None:
        stderr_output = b""
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stderr is not None:
            stderr_output = self._process.stderr.read()
            self._process.stderr.close()
        return_code = self._process.wait()
        if return_code != 0:
            message = stderr_output.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg h264 encoding failed for {self.output_path} "
                f"(exit_code={return_code}): {message}"
            )


def benchmark_device(
    *,
    device_name: str,
    mode: str,
    model_dir: str,
    class_names: list[str],
    frames: list[np.ndarray],
    dtype: str,
    score_threshold: float,
    max_detections: int,
    warmup_frames: int,
    video_meta: dict[str, float | int],
    video_output_dir: Path,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(f"Only CUDA devices are supported in this benchmark, got: {device_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.index is None or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device does not exist: {device_name}")
    device_index = int(device.index)

    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_load = time.perf_counter()
        inferencer = YoloBoxInferencer(
            model_dir=model_dir,
            textencoder_model_dir=args.textencoder_model_dir,
            class_names=class_names,
            device=device,
            dtype=dtype,
            use_cuda_graph=(mode == "cuda_graph"),
        )
        synchronize_device(device)
        load_time_s = time.perf_counter() - start_load

    warmup_count = min(max(0, int(warmup_frames)), len(frames))
    warmup_latencies_s: list[float] = []
    measured_latencies_s: list[float] = []
    detection_counts: list[int] = []
    output_video_path = video_output_dir / f"{device_name.replace(':', '_')}-{mode}.mp4"
    writer = H264VideoWriter(
        output_video_path,
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
        fps=float(video_meta["video_fps"]) if float(video_meta["video_fps"]) > 0 else 30.0,
    )

    total_start = time.perf_counter()
    try:
        with torch.cuda.device(device_index):
            for frame_index, frame in enumerate(frames):
                start = time.perf_counter()
                detections = inferencer.infer(
                    bgr=frame,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )
                synchronize_device(device)
                elapsed = time.perf_counter() - start
                detection_counts.append(len(detections))
                if frame_index < warmup_count:
                    warmup_latencies_s.append(elapsed)
                else:
                    measured_latencies_s.append(elapsed)

                writer.write(draw_boxes(frame, detections))
            synchronize_device(device)
            total_elapsed_s = time.perf_counter() - total_start
    except Exception:
        if output_video_path.exists():
            output_video_path.unlink()
        raise
    finally:
        writer.release()

    measured_frames = len(measured_latencies_s)
    measured_total_s = float(sum(measured_latencies_s))
    avg_latency_ms = round(mean(measured_latencies_s) * 1000.0, 3) if measured_latencies_s else None
    throughput_fps = round(measured_frames / measured_total_s, 3) if measured_total_s > 0 else None
    end_to_end_fps = round(len(frames) / total_elapsed_s, 3) if total_elapsed_s > 0 else None

    return {
        "device": device_name,
        "mode": mode,
        "cuda_graph": inferencer.use_cuda_graph,
        "torch_device_name": torch.cuda.get_device_name(device_index),
        "dtype": dtype,
        "model_dir": model_dir,
        "textencoder_model_dir": str(args.textencoder_model_dir),
        "output_video": str(output_video_path),
        "load_time_s": round(load_time_s, 4),
        "warmup_frames": warmup_count,
        "measured_frames": measured_frames,
        "total_frames": len(frames),
        "avg_infer_ms": avg_latency_ms,
        "p50_infer_ms": percentile_ms(measured_latencies_s, 50),
        "p95_infer_ms": percentile_ms(measured_latencies_s, 95),
        "steady_fps": throughput_fps,
        "end_to_end_fps": end_to_end_fps,
        "avg_detections_per_frame": round(float(mean(detection_counts)), 3) if detection_counts else 0.0,
        "peak_memory_mb": round(float(torch.cuda.max_memory_allocated(device_index) / (1024**2)), 1),
    }


def run_benchmark_subprocess(
    *,
    script_path: Path,
    args: argparse.Namespace,
    device_name: str,
    mode: str,
    video_output_dir: Path,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(prefix="speed01-", suffix=".json", delete=False) as tmp:
        worker_json_path = Path(tmp.name)

    command = [
        sys.executable,
        str(script_path),
        "--model-dir",
        str(args.model_dir),
        "--textencoder-model-dir",
        str(args.textencoder_model_dir),
        "--class-names-file",
        str(args.class_names_file),
        "--video",
        str(args.video),
        "--devices",
        device_name,
        "--modes",
        mode,
        "--dtype",
        str(args.dtype),
        "--score-threshold",
        str(args.score_threshold),
        "--max-detections",
        str(args.max_detections),
        "--warmup-frames",
        str(args.warmup_frames),
        "--max-frames",
        str(args.max_frames),
        "--output-root",
        str(args.output_root),
        "--video-output-root",
        str(args.video_output_root),
        "--existing-video-output-dir",
        str(video_output_dir),
        "--worker-json",
        str(worker_json_path),
    ]

    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "device": device_name,
            "mode": mode,
            "cuda_graph": mode == "cuda_graph",
            "status": "failed",
            "error_type": "SubprocessError",
            "error_message": completed.stderr.strip() or completed.stdout.strip() or "benchmark subprocess failed",
        }

    return json.loads(worker_json_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    class_names = parse_class_names(args.class_names_file)
    video_path = Path(args.video).expanduser()
    frames, video_meta = decode_video_frames(video_path, max_frames=max(0, int(args.max_frames)))

    if args.worker_json:
        if len(args.devices) != 1 or len(args.modes) != 1:
            raise ValueError("worker mode expects exactly one device and one mode.")
        video_output_dir = Path(args.existing_video_output_dir).expanduser()
        result = benchmark_device(
            device_name=args.devices[0],
            mode=args.modes[0],
            model_dir=args.model_dir,
            class_names=class_names,
            frames=frames,
            dtype=args.dtype,
            score_threshold=float(args.score_threshold),
            max_detections=int(args.max_detections),
            warmup_frames=int(args.warmup_frames),
            video_meta=video_meta,
            video_output_dir=video_output_dir,
        )
        Path(args.worker_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    report_path = build_report_path(args.output_root)
    video_output_dir = (
        Path(args.existing_video_output_dir).expanduser()
        if args.existing_video_output_dir
        else build_video_output_dir(args.video_output_root)
    )
    script_path = Path(__file__).resolve()

    results = [
    ]
    for mode in args.modes:
        for device_name in args.devices:
            try:
                result = run_benchmark_subprocess(
                    script_path=script_path,
                    args=args,
                    device_name=device_name,
                    mode=mode,
                    video_output_dir=video_output_dir,
                )
            except Exception as exc:
                result = {
                    "device": device_name,
                    "mode": mode,
                    "cuda_graph": mode == "cuda_graph",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)

    comparisons: list[dict[str, object]] = []
    for device_name in args.devices:
        base_result = next((item for item in results if item["device"] == device_name and item["mode"] == "base"), None)
        graph_result = next(
            (item for item in results if item["device"] == device_name and item["mode"] == "cuda_graph"),
            None,
        )
        if not base_result or not graph_result:
            continue
        if base_result.get("status") == "failed" or graph_result.get("status") == "failed":
            comparisons.append(
                {
                    "device": device_name,
                    "status": "incomplete",
                    "base_status": base_result.get("status", "ok"),
                    "cuda_graph_status": graph_result.get("status", "ok"),
                }
            )
            continue

        base_avg = base_result["avg_infer_ms"]
        graph_avg = graph_result["avg_infer_ms"]
        base_fps = base_result["steady_fps"]
        graph_fps = graph_result["steady_fps"]
        avg_speedup = None
        fps_speedup = None
        if isinstance(base_avg, (int, float)) and isinstance(graph_avg, (int, float)) and graph_avg > 0:
            avg_speedup = round(float(base_avg) / float(graph_avg), 3)
        if isinstance(base_fps, (int, float)) and isinstance(graph_fps, (int, float)) and base_fps > 0:
            fps_speedup = round(float(graph_fps) / float(base_fps), 3)

        comparisons.append(
            {
                "device": device_name,
                "base_avg_infer_ms": base_avg,
                "cuda_graph_avg_infer_ms": graph_avg,
                "avg_infer_ms_speedup": avg_speedup,
                "base_steady_fps": base_fps,
                "cuda_graph_steady_fps": graph_fps,
                "steady_fps_speedup": fps_speedup,
            }
        )

    report = {
        "video": str(video_path),
        "video_meta": video_meta,
        "class_names": class_names,
        "results": results,
        "comparisons": comparisons,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report: {report_path}")
    print(f"video_output_dir: {video_output_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
