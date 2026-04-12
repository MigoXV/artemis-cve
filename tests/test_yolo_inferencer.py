from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from artemis_cve.inferencers.yolo.inferencer import YoloBoxInferencer
from artemis_cve.inferencers.yolo.runtime import BaseYoloRuntime, RawYoloOutput
from artemis_cve.models.yolo26e import YOLOEConfig, YOLOEModel


class _FakeRuntime:
    def __init__(self, output: RawYoloOutput) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs) -> RawYoloOutput:
        self.calls.append(kwargs)
        return self.output


class _FakeTextEncoder:
    def encode(self, texts, *, dtype=torch.float32):
        del texts, dtype
        return torch.ones((1, 1, 4), dtype=torch.float32)


class YoloBoxInferencerTests(unittest.TestCase):
    def test_infer_filters_and_limits_detections(self) -> None:
        inferencer = object.__new__(YoloBoxInferencer)
        inferencer.device = torch.device("cpu")
        inferencer.dtype = torch.float32
        inferencer.class_names = ("car", "person")
        inferencer.config = YOLOEConfig(
            default_classes=["car", "person"],
            open_vocab=False,
            id2label={0: "car", 1: "person"},
        )
        inferencer._letterbox = lambda image: image
        inferencer.text_embeddings = torch.ones((1, 2, 4), dtype=torch.float32)
        inferencer.runtime = _FakeRuntime(
            RawYoloOutput(
                boxes=[torch.tensor([[1, 1, 4, 4], [0, 0, 2, 2]], dtype=torch.float32)],
                scores=[torch.tensor([0.95, 0.2], dtype=torch.float32)],
                labels=[torch.tensor([1, 0], dtype=torch.long)],
                masks=[None],
                raw_results=None,
            )
        )

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = inferencer.infer(
            bgr=image,
            score_threshold=0.5,
            max_detections=1,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[0].pixel_xyxy, (1.0, 1.0, 4.0, 4.0))
        self.assertEqual(
            inferencer.runtime.calls[0]["max_det"],
            1,
        )
        self.assertFalse(inferencer.runtime.calls[0]["include_masks"])

    def test_convert_outputs_uses_config_fallback_class_name(self) -> None:
        inferencer = object.__new__(YoloBoxInferencer)
        inferencer.class_names = ("car",)
        inferencer.text_embeddings = torch.ones((1, 1, 4), dtype=torch.float32)
        inferencer.config = YOLOEConfig(
            default_classes=["car"],
            open_vocab=False,
            id2label={"5": "bus"},
        )

        detections = inferencer._convert_outputs(
            outputs=RawYoloOutput(
                boxes=[torch.tensor([[2, 2, 6, 6]], dtype=torch.float32)],
                scores=[torch.tensor([0.8], dtype=torch.float32)],
                labels=[torch.tensor([5], dtype=torch.long)],
                masks=[None],
                raw_results=None,
            ),
            original_shape=(10, 10),
            processed_shape=(10, 10),
            score_threshold=0.0,
            max_detections=None,
        )

        self.assertEqual(detections[0].class_name, "bus")
        self.assertEqual(detections[0].normalized_xyxy, (0.2, 0.2, 0.6, 0.6))

    def test_init_rejects_cuda_graph_on_cpu(self) -> None:
        config = YOLOEConfig(default_classes=["car"], open_vocab=False)
        model = YOLOEModel(config)

        with patch(
            "artemis_cve.inferencers.yolo.inferencer.AutoConfig.from_pretrained",
            return_value=config,
        ), patch(
            "artemis_cve.inferencers.yolo.inferencer.AutoModel.from_pretrained",
            return_value=model,
        ), patch(
            "artemis_cve.inferencers.yolo.inferencer.YOLOETextEncoder.from_pretrained",
            return_value=_FakeTextEncoder(),
        ):
            with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
                YoloBoxInferencer(
                    model_dir="unused",
                    textencoder_model_dir="unused-text",
                    class_names=["car"],
                    device="cpu",
                    use_cuda_graph=True,
                )

    def test_init_uses_base_runtime_by_default(self) -> None:
        config = YOLOEConfig(default_classes=["car"], open_vocab=False)
        model = YOLOEModel(config)

        with patch(
            "artemis_cve.inferencers.yolo.inferencer.AutoConfig.from_pretrained",
            return_value=config,
        ), patch(
            "artemis_cve.inferencers.yolo.inferencer.AutoModel.from_pretrained",
            return_value=model,
        ), patch(
            "artemis_cve.inferencers.yolo.inferencer.YOLOETextEncoder.from_pretrained",
            return_value=_FakeTextEncoder(),
        ):
            inferencer = YoloBoxInferencer(
                model_dir="unused",
                textencoder_model_dir="unused-text",
                class_names=["car"],
                device="cpu",
                use_cuda_graph=False,
            )

        self.assertIsInstance(inferencer.runtime, BaseYoloRuntime)
