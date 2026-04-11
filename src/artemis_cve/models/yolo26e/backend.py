from __future__ import annotations

from typing import Any

import torch
from ultralytics.nn.modules import (
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    RepConv,
    RepVGGDW,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
)
from ultralytics.nn.tasks import YOLOEModel as UltralyticsYOLOEModel
from ultralytics.nn.tasks import YOLOESegModel as UltralyticsYOLOESegModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn

from .configuration import YOLOEConfig
from .outputs import YOLOERawOutput


def build_ultralytics_task_model(config: YOLOEConfig) -> torch.nn.Module:
    """根据配置构建 Ultralytics YOLOE 任务模型。

    Args:
        config: YOLOE 配置对象。

    Returns:
        已初始化完成的 Ultralytics 模型实例；当 `config.fused=True` 时，
        返回融合后的版本。
    """

    task_name = str(config.task or "").strip().lower()
    is_segmentation = bool(config.segmentation) or "seg" in str(config.variant).lower() or "segment" in task_name
    model_cls = UltralyticsYOLOESegModel if is_segmentation else UltralyticsYOLOEModel
    num_labels = int(config.num_labels) if int(config.num_labels or 0) > 0 else None
    task_model = model_cls(
        cfg=f"{config.variant}.yaml",
        ch=int(config.num_channels),
        nc=num_labels,
        verbose=False,
    )
    if bool(getattr(config, "fused", False)):
        task_model = fuse_task_model_preserving_head(task_model)
    return task_model


def fuse_task_model_preserving_head(task_model: torch.nn.Module) -> torch.nn.Module:
    """融合主干中的卷积与归一化层，同时保留检测头结构。

    Ultralytics 的通用 `fuse()` 在某些 YOLOE 路径上会改变 head 行为，
    因此这里手动遍历模块，只对安全的卷积块执行融合。
    """

    if task_model.is_fused():
        return task_model

    for module in task_model.model.modules():
        # 仅处理推理期可安全折叠的卷积结构，避免破坏 YOLOE head 的特殊逻辑。
        if isinstance(module, (Conv, Conv2, DWConv)) and hasattr(module, "bn"):
            if isinstance(module, Conv2):
                module.fuse_convs()
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
        elif isinstance(module, ConvTranspose) and hasattr(module, "bn"):
            module.conv_transpose = fuse_deconv_and_bn(module.conv_transpose, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
        elif isinstance(module, RepConv):
            module.fuse_convs()
            module.forward = module.forward_fuse
        elif isinstance(module, RepVGGDW):
            module.fuse()
            module.forward = module.forward_fuse

    return task_model


def normalize_text_embeddings(
    text_embeddings: torch.Tensor,
    *,
    expected_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """校验并规范化文本嵌入张量。

    Args:
        text_embeddings: 原始文本嵌入，支持 `[classes, dim]` 或 `[batch, classes, dim]`。
        expected_dim: 期望的嵌入维度。
        device: 目标设备。
        dtype: 目标数据类型。

    Returns:
        已扩展到三维、移动到目标设备并复制后的嵌入张量。

    Raises:
        TypeError: 输入不是张量时抛出。
        ValueError: 输入形状或末维不符合预期时抛出。
    """

    if not isinstance(text_embeddings, torch.Tensor):
        raise TypeError(f"text_embeddings must be a torch.Tensor, got {type(text_embeddings)!r}")
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(0)
    if text_embeddings.ndim != 3:
        raise ValueError(
            "text_embeddings must have shape [batch, classes, embedding_dim] or [classes, embedding_dim]. "
            f"Received shape: {tuple(text_embeddings.shape)}"
        )
    if text_embeddings.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected text embedding dim {expected_dim}, received {text_embeddings.shape[-1]}."
        )
    return text_embeddings.to(device=device, dtype=dtype).clone()


def forward_yoloe_head_raw(
    head: YOLOEDetect,
    feature_maps: list[torch.Tensor],
    cls_embeddings: torch.Tensor,
    *,
    include_masks: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...] | None]:
    """直接调用 YOLOE 检测头，返回未后处理的预测结果。

    Args:
        head: YOLOE 检测头模块。
        feature_maps: 多尺度特征图列表。
        cls_embeddings: 已投影到 head 维度的类别嵌入。
        include_masks: 是否同时计算分割相关输出。

    Returns:
        一个二元组：
        - 第一个元素为 head 返回的原始预测字典。
        - 第二个元素为可选的分割 prototype。
    """

    head_kwargs = dict(head.one2one if head.end2end else head.one2many)
    if not include_masks and "mask_head" in head_kwargs:
        head_kwargs["mask_head"] = None
    if head.end2end:
        head_inputs = [tensor.detach() for tensor in feature_maps] + [cls_embeddings.detach()]
        preds = head.forward_head(head_inputs, **head_kwargs)
    else:
        head_inputs = list(feature_maps) + [cls_embeddings]
        preds = head.forward_head(head_inputs, **head_kwargs)

    prototypes = None
    if include_masks and isinstance(head, YOLOESegment26):
        prototypes = head.proto([tensor.detach() for tensor in feature_maps], return_semseg=False)
    elif include_masks and isinstance(head, YOLOESegment):
        prototypes = head.proto(feature_maps[0])
    return preds, prototypes


def forward_yoloe_task_model_raw(
    task_model: Any,
    pixel_values: torch.Tensor,
    *,
    text_embeddings: torch.Tensor | None = None,
    return_text_embeddings: bool = False,
    include_masks: bool = True,
) -> YOLOERawOutput:
    """执行 YOLOE 模型的原始前向过程。

    该函数绕过高层封装，逐层运行 Ultralytics 模型，直到命中检测头，
    然后返回未经后处理的框、分类和可选分割输出。

    Args:
        task_model: 拥有 `model`、`save` 与 `get_cls_pe` 等接口的 YOLOE 模型对象。
        pixel_values: 输入图像张量，形状应为 `[batch, channels, height, width]`。
        text_embeddings: 可选文本嵌入；开放词表模型通常必须提供。
        return_text_embeddings: 是否在输出中回传规范化后的文本嵌入。
        include_masks: 是否计算分割相关输出。

    Returns:
        `YOLOERawOutput`，包含 head 原始输出与必要元数据。

    Raises:
        TypeError: 输入张量类型不正确，或检测头类型不符合预期时抛出。
        ValueError: 输入形状不正确，或开放词表场景缺少文本嵌入时抛出。
        RuntimeError: 遍历模型后仍未命中检测头时抛出。
    """

    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError(f"pixel_values must be a torch.Tensor, got {type(pixel_values)!r}")
    if pixel_values.ndim != 4:
        raise ValueError(
            "pixel_values must have shape [batch, channels, height, width]. "
            f"Received shape: {tuple(pixel_values.shape)}"
        )

    head = task_model.model[-1]
    if not isinstance(head, YOLOEDetect):
        raise TypeError(f"Expected a YOLOEDetect-compatible head, got {type(head)!r}")

    if text_embeddings is None and not hasattr(head, "lrpc"):
        raise ValueError("text_embeddings must be provided for this model.")

    normalized_text_embeddings = None
    if text_embeddings is not None:
        normalized_text_embeddings = normalize_text_embeddings(
            text_embeddings,
            expected_dim=int(getattr(task_model.config, "text_embedding_dim", 512))
            if hasattr(task_model, "config")
            else 512,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

    outputs: list[torch.Tensor | None] = []
    x: torch.Tensor | list[torch.Tensor] = pixel_values
    batch_size = int(pixel_values.shape[0])

    for module in task_model.model:
        if module.f != -1:
            x = outputs[module.f] if isinstance(module.f, int) else [x if idx == -1 else outputs[idx] for idx in module.f]

        if isinstance(module, YOLOEDetect):
            assert isinstance(x, list)
            feature_maps = list(x[:3])
            if normalized_text_embeddings is None:
                cls_embeddings = task_model.get_cls_pe(None, None).to(
                    device=feature_maps[0].device,
                    dtype=feature_maps[0].dtype,
                )
            else:
                projected_embeddings = module.get_tpe(normalized_text_embeddings)
                cls_embeddings = task_model.get_cls_pe(projected_embeddings, None).to(
                    device=feature_maps[0].device,
                    dtype=feature_maps[0].dtype,
                )
                if cls_embeddings.shape[0] != batch_size or bool(getattr(module, "export", False)):
                    # 导出模式或单批嵌入场景下，需要按 batch 维展开类别嵌入。
                    cls_embeddings = cls_embeddings.expand(batch_size, -1, -1)

            preds, prototypes = forward_yoloe_head_raw(
                module,
                feature_maps,
                cls_embeddings,
                include_masks=include_masks,
            )
            return YOLOERawOutput(
                box_logits=preds["boxes"],
                class_logits=preds["scores"],
                mask_coefficients=preds.get("mask_coefficient") if include_masks else None,
                prototypes=prototypes,
                feature_shapes=tuple((int(feat.shape[2]), int(feat.shape[3])) for feat in feature_maps),
                strides=tuple(int(value) for value in module.stride.tolist()),
                text_embeddings=normalized_text_embeddings if return_text_embeddings else None,
            )

        x = module(x)
        outputs.append(x if module.i in task_model.save else None)

    raise RuntimeError("Failed to locate the YOLOE detection head during raw forward.")
