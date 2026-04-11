from __future__ import annotations

import unittest

import torch

from artemis_cve.inferencers.yolo.runtime import BaseYoloRuntime, CudaGraphYoloRuntime
from artemis_cve.models.yolo26e import YOLOEConfig, YOLOERawOutput


class _FakeHead:
    def __init__(self, *, end2end: bool = True, nc: int = 2) -> None:
        self.end2end = end2end
        self.nc = nc
        self.dfl = torch.nn.Identity()

    def decode_bboxes(self, boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        del anchors
        return boxes

    def get_topk_index(self, scores: torch.Tensor, max_det: int):
        batch_size, anchors, nc = scores.shape
        k = min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        gathered_scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        top_scores, flat_index = gathered_scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], flat_index // nc]
        return top_scores[..., None], (flat_index % nc)[..., None].float(), idx


class _FakeModel:
    def __init__(self, output: YOLOERawOutput, *, end2end: bool = True) -> None:
        self.output = output
        self.config = YOLOEConfig(
            default_classes=["car", "person"],
            open_vocab=True,
            text_embedding_dim=4,
            score_threshold=0.25,
            iou_threshold=0.7,
        )
        self._head = _FakeHead(end2end=end2end, nc=2)
        self.forward_calls: list[dict[str, object]] = []

    def get_detection_head(self) -> _FakeHead:
        return self._head

    def __call__(
        self,
        *,
        pixel_values: torch.Tensor,
        text_embeddings: torch.Tensor,
        include_masks: bool = True,
    ) -> YOLOERawOutput:
        self.forward_calls.append(
            {
                "pixel_values_shape": tuple(pixel_values.shape),
                "text_embeddings_shape": tuple(text_embeddings.shape),
                "include_masks": include_masks,
            }
        )
        return self.output


class BaseYoloRuntimeTests(unittest.TestCase):
    def test_forward_decodes_topk_predictions(self) -> None:
        model = _FakeModel(
            YOLOERawOutput(
                box_logits=torch.tensor(
                    [
                        [
                            [1.0, 3.0, 5.0],
                            [1.0, 3.0, 5.0],
                            [2.0, 4.0, 6.0],
                            [2.0, 4.0, 6.0],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                class_logits=torch.tensor(
                    [
                        [
                            [10.0, -10.0, -10.0],
                            [-10.0, 10.0, -10.0],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                feature_shapes=((1, 3),),
                strides=(1,),
            )
        )
        runtime = BaseYoloRuntime(model)

        output = runtime.forward(
            pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float16),
            text_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
            max_det=2,
        )

        self.assertEqual(len(model.forward_calls), 1)
        self.assertTrue(model.forward_calls[0]["include_masks"])
        self.assertEqual(tuple(output.boxes[0].shape), (2, 4))
        self.assertEqual(sorted(output.labels[0].tolist()), [0, 1])
        self.assertEqual(
            sorted(tuple(box) for box in output.boxes[0].tolist()),
            [(1.0, 1.0, 2.0, 2.0), (3.0, 3.0, 4.0, 4.0)],
        )
        self.assertTrue(torch.all(output.scores[0] > 0.99))

    def test_forward_requires_valid_text_embeddings(self) -> None:
        model = _FakeModel(
            YOLOERawOutput(
                box_logits=torch.zeros((1, 4, 1), dtype=torch.float32),
                class_logits=torch.zeros((1, 2, 1), dtype=torch.float32),
                feature_shapes=((1, 1),),
                strides=(1,),
            )
        )
        runtime = BaseYoloRuntime(model)

        with self.assertRaisesRegex(ValueError, "text_embeddings must have shape"):
            runtime.forward(
                pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float32),
                text_embeddings=torch.ones((1, 2, 4, 1), dtype=torch.float32),
            )


class CudaGraphYoloRuntimeTests(unittest.TestCase):
    def test_forward_on_cpu_uses_raw_model_without_cuda_graph(self) -> None:
        model = _FakeModel(
            YOLOERawOutput(
                box_logits=torch.tensor([[[1.0], [1.0], [2.0], [2.0]]], dtype=torch.float32),
                class_logits=torch.tensor([[[10.0], [-10.0]]], dtype=torch.float32),
                feature_shapes=((1, 1),),
                strides=(1,),
            )
        )
        runtime = CudaGraphYoloRuntime(model)

        output = runtime.forward(
            pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float32),
            text_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
            max_det=1,
        )

        self.assertEqual(len(model.forward_calls), 1)
        self.assertEqual(tuple(output.boxes[0].shape), (1, 4))
        self.assertEqual(int(output.labels[0][0].item()), 0)

    def test_forward_can_disable_mask_outputs(self) -> None:
        model = _FakeModel(
            YOLOERawOutput(
                box_logits=torch.tensor([[[1.0], [1.0], [2.0], [2.0]]], dtype=torch.float32),
                class_logits=torch.tensor([[[10.0], [-10.0]]], dtype=torch.float32),
                mask_coefficients=torch.tensor([[[0.5]]], dtype=torch.float32),
                feature_shapes=((1, 1),),
                strides=(1,),
            )
        )
        runtime = BaseYoloRuntime(model)

        output = runtime.forward(
            pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float32),
            text_embeddings=torch.ones((1, 2, 4), dtype=torch.float32),
            max_det=1,
            include_masks=False,
        )

        self.assertFalse(model.forward_calls[0]["include_masks"])
        self.assertEqual(output.masks, [None])
