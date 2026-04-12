from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from artemis_cve.inferencers.yolo import YoloBoxInferencer

DEFAULT_MODEL_DIR = Path("model-bin/MigoXV/yoloe26-x-seg")
DEFAULT_TEXT_ENCODER_MODEL_DIR = Path("model-bin/MigoXV/mobileclip2-b")
DEFAULT_OUTPUT_DIR = Path("data-bin/demo-outputs")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run artemis-cve box inferencer on local images.")
    parser.add_argument("images", nargs="+", help="Input image paths.")
    parser.add_argument(
        "--class-names",
        required=True,
        help="Comma-separated class names for open-vocabulary inference.",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--textencoder-model-dir", default=str(DEFAULT_TEXT_ENCODER_MODEL_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    inferencer = YoloBoxInferencer(
        model_dir=args.model_dir,
        textencoder_model_dir=args.textencoder_model_dir,
        class_names=class_names,
        device=args.device,
        dtype=args.dtype,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        detections = inferencer.infer(
            bgr=bgr,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )
        print(f"{image_path} -> {len(detections)} detections")
        for idx, detection in enumerate(detections):
            print(
                f"  {idx}: class={detection.class_name} "
                f"score={detection.score:.4f} "
                f"box={[round(v, 2) for v in detection.pixel_xyxy]}"
            )

        vis = draw_boxes(bgr, detections)
        output_path = output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), vis)
        print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
