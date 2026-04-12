from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from artemis_cve.inferencers.yolo import YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_CLASS_NAMES_FILE = Path("model-bin/names.txt")
DEFAULT_IMAGE_DIR = Path("data-bin/images")
DEFAULT_OUTPUT_ROOT = Path("data-bin/outputs")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def collect_images(input_dir: str) -> list[Path]:
    image_dir = Path(input_dir).expanduser()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"input image directory does not exist: {image_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no supported image files found in: {image_dir}")
    return images


def draw_boxes(image, detections):
    vis = image.copy()
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = [int(round(value)) for value in detection.pixel_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{idx}: {detection.class_name} {detection.score:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return vis


def build_output_dir(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_root).expanduser() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOE inferencer on images and save visualized outputs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_IMAGE_DIR), help="Directory containing input images.")
    parser.add_argument("--class-names-file", default=str(DEFAULT_CLASS_NAMES_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--textencoder-model-dir", default=str(DEFAULT_TEXT_ENCODER_MODEL_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)
    class_names = parse_class_names(args.class_names_file)
    output_dir = build_output_dir(args.output_root)

    inferencer = YoloBoxInferencer(
        model_dir=args.model_dir,
        textencoder_model_dir=args.textencoder_model_dir,
        class_names=class_names,
        device=args.device,
        dtype=args.dtype,
    )

    print(f"input_dir: {Path(args.input_dir).expanduser()}")
    print(f"output_dir: {output_dir}")
    print(f"class_names: {class_names}")

    for image_path in image_paths:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")

        detections = inferencer.infer(
            bgr=bgr,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )
        print(f"{image_path.name} -> {len(detections)} detections")
        for idx, detection in enumerate(detections):
            print(
                f"  {idx}: class={detection.class_name} "
                f"score={detection.score:.4f} "
                f"box={[round(v, 2) for v in detection.pixel_xyxy]}"
            )

        vis = draw_boxes(bgr, detections)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), vis)
        print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
