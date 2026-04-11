from __future__ import annotations

from transformers import PretrainedConfig


class YOLOEConfig(PretrainedConfig):
    """YOLO26E 的 Hugging Face 配置类。

    该类在 `PretrainedConfig` 基础上补充了 YOLOE 所需的结构化字段，
    例如变体名称、分割开关、默认类别、文本编码器资产名等。
    """

    model_type = "yoloe"

    def __init__(
        self,
        variant: str = "yoloe-26x-seg",
        task: str = "instance-segmentation",
        image_size: int = 640,
        num_channels: int = 3,
        segmentation: bool = True,
        fused: bool = False,
        open_vocab: bool = True,
        default_classes: list[str] | None = None,
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        stride: list[int] | None = None,
        model_input_name: str = "pixel_values",
        dtype: str = "float32",
        torch_dtype: str | None = None,
        text_encoder_type: str = "mobileclip2",
        text_encoder_asset: str = "mobileclip2_b.ts",
        text_encoder_path: str | None = None,
        text_embedding_dim: int = 512,
        **kwargs,
    ) -> None:
        """初始化 YOLO26E 配置。

        Args:
            variant: Ultralytics 模型变体名称。
            task: 任务类型描述，如检测或实例分割。
            image_size: 训练/推理默认输入边长。
            num_channels: 输入图像通道数。
            segmentation: 是否启用分割头。
            fused: 是否在构建时执行卷积融合。
            open_vocab: 是否启用开放词表能力。
            default_classes: 默认类别列表。
            num_labels: 类别数量；未提供时会根据标签映射自动推断。
            id2label: 类别 ID 到名称的映射。
            label2id: 类别名称到 ID 的映射。
            score_threshold: 默认置信度阈值。
            iou_threshold: 默认 NMS IoU 阈值。
            stride: 主干输出步长列表。
            model_input_name: 主输入字段名称。
            dtype: 默认数据类型字符串。
            torch_dtype: 可覆盖 `dtype` 的 Torch 数据类型描述。
            text_encoder_type: 文本编码器类型标识。
            text_encoder_asset: 内嵌文本编码器资产名。
            text_encoder_path: 文本编码器所在目录。
            text_embedding_dim: 文本嵌入维度。
            **kwargs: 透传给 `PretrainedConfig` 的附加字段。
        """

        default_classes = default_classes or []
        stride = stride or [8, 16, 32]
        resolved_dtype = torch_dtype or dtype

        if id2label is None:
            id2label = {idx: name for idx, name in enumerate(default_classes)}
        else:
            id2label = {int(idx): name for idx, name in id2label.items()}

        if label2id is None:
            label2id = {name: idx for idx, name in id2label.items()}

        if num_labels is None:
            num_labels = len(id2label)

        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            dtype=resolved_dtype,
            **kwargs,
        )

        # 保存 YOLOE 专有字段，供模型构建与推理过程读取。
        self.variant = variant
        self.task = task
        self.image_size = image_size
        self.num_channels = num_channels
        self.segmentation = segmentation
        self.fused = fused
        self.open_vocab = open_vocab
        self.default_classes = list(default_classes)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.stride = list(stride)
        self.model_input_name = model_input_name
        self.dtype = resolved_dtype
        self.text_encoder_type = text_encoder_type
        self.text_encoder_asset = text_encoder_asset
        self.text_encoder_path = text_encoder_path
        self.text_embedding_dim = int(text_embedding_dim)
