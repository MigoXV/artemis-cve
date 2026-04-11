from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedModel
from ultralytics.nn.modules import YOLOEDetect

from .backend import (
    build_ultralytics_task_model,
    forward_yoloe_task_model_raw,
    normalize_text_embeddings,
)
from .configuration import YOLOEConfig
from .io import load_checkpoint_state, resolve_weights_path
from .outputs import YOLOERawOutput


class YOLOEModel(PreTrainedModel):
    """适配 Hugging Face 接口的 YOLOE 模型。

    模型主体仍然使用 Ultralytics 的内部模块，但对外暴露
    `from_pretrained`、`forward`、配置绑定与权重校验能力，
    方便被 `AutoModel` 和项目推理层统一调用。
    """

    config_class = YOLOEConfig
    base_model_prefix = "yoloe"
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_missing = [r".*"]
    _keys_to_ignore_on_load_unexpected = [r".*"]

    def __init__(self, config: YOLOEConfig) -> None:
        """根据配置构建底层 Ultralytics YOLOE 模型。"""

        super().__init__(config)
        task_model = build_ultralytics_task_model(config)
        self.model = task_model.model
        self.save = list(task_model.save)
        self.stride = task_model.stride
        self.yaml = getattr(task_model, "yaml", {})
        self.args = getattr(task_model, "args", {})
        self.names = getattr(task_model, "names", None)
        self.text_model = getattr(task_model, "text_model", f"{config.text_encoder_type}:b")
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args,
        **kwargs,
    ) -> "YOLOEModel":
        """从导出的模型目录或 Hub 仓库加载 YOLOE 模型。

        与默认的 `PreTrainedModel.from_pretrained` 不同，这里显式读取
        `model.safetensors`，并要求权重键与模型结构严格一致。
        """

        if model_args:
            raise ValueError("YOLOEModel.from_pretrained does not accept positional model args.")

        requested_dtype = kwargs.pop("dtype", None)
        requested_torch_dtype = kwargs.pop("torch_dtype", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        resolved_config = kwargs.pop("config", None)
        if resolved_config is None:
            config_kwargs = dict(kwargs)
            if token is not None:
                config_kwargs["token"] = token
            if revision is not None:
                config_kwargs["revision"] = revision
            resolved_config = cls.config_class.from_pretrained(str(pretrained_model_name_or_path), **config_kwargs)

        model = cls(resolved_config)
        weights_path = resolve_weights_path(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
        )
        if weights_path is None:
            raise FileNotFoundError("Could not find model.safetensors for pretrained loading.")

        state_dict, _ = load_checkpoint_state(weights_path)
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict checkpoint loading failed. "
                f"missing={incompatible.missing_keys[:10]} unexpected={incompatible.unexpected_keys[:10]}"
            )
        model.validate_checkpoint_keys(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
        )

        target_dtype = requested_torch_dtype or requested_dtype
        if target_dtype is not None:
            model = model.to(dtype=target_dtype)
        return model

    def validate_checkpoint_keys(
        self,
        pretrained_model_name_or_path: str | Path | None = None,
        *,
        token: str | None = None,
        revision: str | None = None,
    ) -> None:
        """校验 checkpoint 键集合是否与当前模型完全一致。

        该校验用于尽早发现导出布局与运行时代码不匹配的问题，
        比如模型结构升级后仍误用旧权重。
        """

        source = pretrained_model_name_or_path or self.config.name_or_path
        weights_path = resolve_weights_path(
            source,
            token=token,
            revision=revision,
        )
        if weights_path is None:
            raise FileNotFoundError("Could not find model.safetensors for checkpoint validation.")

        checkpoint_state, _ = load_checkpoint_state(weights_path)
        checkpoint_keys = set(checkpoint_state)
        model_keys = set(self.state_dict())
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint keys do not match the pure nn.Module model layout. "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    def get_detection_head(self) -> YOLOEDetect:
        """返回模型末端的 YOLOE 检测头。"""

        head = self.model[-1]
        if not isinstance(head, YOLOEDetect):
            raise TypeError(f"Expected YOLOEDetect-compatible head, got {type(head)!r}")
        return head

    def get_cls_pe(
        self,
        text_prompt_embeddings: torch.Tensor | None,
        visual_prompt_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """拼接文本提示嵌入与视觉提示嵌入。

        YOLOE 的提示编码接口统一把不同来源的 prompt 在类别维拼接，
        因此这里提供一个最小封装，避免调用方重复处理空值逻辑。
        """

        all_embeddings: list[torch.Tensor] = []
        if text_prompt_embeddings is not None:
            all_embeddings.append(text_prompt_embeddings)
        if visual_prompt_embeddings is not None:
            all_embeddings.append(visual_prompt_embeddings)
        if not all_embeddings:
            raise ValueError("At least one prompt embedding tensor is required.")
        return torch.cat(all_embeddings, dim=1)

    def project_text_embeddings(
        self,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """把外部文本嵌入投影到检测头使用的提示空间。"""

        head = self.get_detection_head()
        normalized = normalize_text_embeddings(
            text_embeddings,
            expected_dim=int(self.config.text_embedding_dim),
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        return head.get_tpe(normalized)

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        text_embeddings: torch.Tensor | None = None,
        return_text_embeddings: bool = False,
        include_masks: bool = True,
    ) -> YOLOERawOutput:
        """执行模型前向，并返回未后处理的 YOLOE 输出。

        Args:
            pixel_values: 输入图像张量。
            text_embeddings: 可选文本嵌入。
            return_text_embeddings: 是否回传规范化后的文本嵌入。
            include_masks: 是否包含分割相关输出。
        """

        if pixel_values.shape[1] != self.config.num_channels:
            raise ValueError(
                f"Expected {self.config.num_channels} channels, received {pixel_values.shape[1]}."
            )
        if not pixel_values.is_floating_point():
            pixel_values = pixel_values.float()

        return forward_yoloe_task_model_raw(
            self,
            pixel_values,
            text_embeddings=text_embeddings,
            return_text_embeddings=return_text_embeddings,
            include_masks=include_masks,
        )
