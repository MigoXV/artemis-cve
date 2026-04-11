from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig, AutoModel, AutoProcessor

from .yolo26e import YOLOEConfig, YOLOEModel


@dataclass(frozen=True, slots=True)
class TransformersRegistration:
    """描述一次 Transformers 自动工厂注册所需的元数据。

    Attributes:
        model_type: `AutoConfig` 查找时使用的模型类型字符串。
        config_class: 与模型类型对应的配置类。
        model_class: 与配置类对应的模型类。
        processor_class: 可选的处理器类；当前仓库暂无对应实现。
    """

    model_type: str
    config_class: type
    model_class: type
    processor_class: type | None = None


_REGISTERED = False
_REGISTRATIONS = (
    TransformersRegistration(
        model_type=YOLOEConfig.model_type,
        config_class=YOLOEConfig,
        model_class=YOLOEModel,
    ),
)


def ensure_model_registrations() -> None:
    """确保自定义模型仅注册一次。

    该函数被包入口调用，也可以被外部显式调用。重复调用是安全的；
    一旦 `_REGISTERED` 置位，后续调用会直接返回。
    """

    global _REGISTERED

    if _REGISTERED:
        return

    for registration in _REGISTRATIONS:
        # 统一把配置类与模型类注册到 Transformers 自动工厂。
        AutoConfig.register(
            registration.model_type,
            registration.config_class,
            exist_ok=True,
        )
        AutoModel.register(
            registration.config_class,
            registration.model_class,
            exist_ok=True,
        )
        if registration.processor_class is not None:
            AutoProcessor.register(
                registration.config_class,
                registration.processor_class,
                exist_ok=True,
            )

    _REGISTERED = True
