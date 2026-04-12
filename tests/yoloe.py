from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from artemis_cve.inferencers.yolo import YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the local YOLOE wrapper.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("--class-names", required=True, help="Comma-separated class names.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--textencoder-model-dir", default=str(DEFAULT_TEXT_ENCODER_MODEL_DIR))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    inferencer = YoloBoxInferencer(
        model_dir=args.model_dir,
        textencoder_model_dir=args.textencoder_model_dir,
        class_names=[item.strip() for item in args.class_names.split(",") if item.strip()],
        device=args.device,
        dtype=args.dtype,
    )
    detections = inferencer.infer(
        bgr=bgr,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    print(f"{args.image} -> {len(detections)} detections")
    for idx, detection in enumerate(detections):
        print(
            f"  {idx}: class={detection.class_name} "
            f"score={detection.score:.4f} "
            f"pixel={[round(v, 2) for v in detection.pixel_xyxy]} "
            f"normalized={[round(v, 4) for v in detection.normalized_xyxy]}"
        )


if __name__ == "__main__":
    main()
