from __future__ import annotations

import tempfile
import unittest

import torch
from safetensors.torch import save_file
from transformers import AutoModel

import artemis_cve.models
from artemis_cve.models.yolo26e import YOLOEConfig, YOLOEModel, YOLOERawOutput


class YoloModelingTests(unittest.TestCase):
    def test_auto_model_from_pretrained_loads_yoloe_model(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            model = YOLOEModel(
                YOLOEConfig(
                    default_classes=["car"],
                    open_vocab=False,
                    num_labels=1,
                )
            )
            model.config.save_pretrained(model_dir)
            save_file(model.state_dict(), f"{model_dir}/model.safetensors")

            loaded = AutoModel.from_pretrained(model_dir)

        self.assertIsInstance(loaded, YOLOEModel)

    def test_forward_returns_raw_output(self) -> None:
        model = YOLOEModel(
            YOLOEConfig(
                default_classes=["car"],
                open_vocab=True,
                text_embedding_dim=512,
            )
        )

        outputs = model(
            torch.ones((1, 3, 64, 64), dtype=torch.float32),
            text_embeddings=torch.ones((1, 1, 512), dtype=torch.float32),
        )

        self.assertIsInstance(outputs, YOLOERawOutput)
        self.assertEqual(outputs.box_logits.ndim, 3)
        self.assertEqual(outputs.class_logits.ndim, 3)
        self.assertTrue(outputs.feature_shapes)

    def test_forward_requires_text_embeddings(self) -> None:
        model = YOLOEModel(YOLOEConfig(default_classes=["car"], open_vocab=True))

        with self.assertRaisesRegex(ValueError, "text_embeddings must be provided"):
            model(torch.ones((1, 3, 64, 64), dtype=torch.float32))

    def test_forward_can_skip_segmentation_outputs(self) -> None:
        model = YOLOEModel(
            YOLOEConfig(
                default_classes=["car"],
                open_vocab=True,
                text_embedding_dim=512,
                segmentation=True,
            )
        )

        outputs = model(
            torch.ones((1, 3, 64, 64), dtype=torch.float32),
            text_embeddings=torch.ones((1, 1, 512), dtype=torch.float32),
            include_masks=False,
        )

        self.assertIsNone(outputs.mask_coefficients)
        self.assertIsNone(outputs.prototypes)
