from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class YOLOERawOutput(ModelOutput):
    """承载 YOLOE head 原始输出的结构化对象。

    Attributes:
        box_logits: 原始框回归输出。
        class_logits: 原始分类得分输出。
        mask_coefficients: 分割场景下的掩码系数。
        prototypes: 分割原型特征。
        feature_shapes: 进入检测头的多尺度特征尺寸。
        strides: 各特征层步长。
        text_embeddings: 可选回传的归一化文本嵌入。
    """

    box_logits: torch.Tensor | None = None
    class_logits: torch.Tensor | None = None
    mask_coefficients: torch.Tensor | None = None
    prototypes: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    feature_shapes: tuple[tuple[int, int], ...] | None = None
    strides: tuple[int, ...] | None = None
    text_embeddings: torch.Tensor | None = None
