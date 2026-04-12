from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import clip
import torch
from transformers import AutoConfig

from .io import ASSET_PREFIX, load_checkpoint_state, resolve_weights_path


class MobileCLIPTextEncoder:
    """基于 TorchScript 的 MobileCLIP2 文本编码器封装。

    该类提供三个层次的接口：
    1. `tokenize`：将字符串提示词转为 token。
    2. `encode_tokens`：直接编码 token 张量。
    3. `encode`：一站式完成文本清洗、分词与编码。
    """

    def __init__(
        self,
        *,
        encoder: torch.jit.ScriptModule,
        device: torch.device,
    ) -> None:
        """初始化文本编码器实例。

        Args:
            encoder: 已加载的 TorchScript 编码器。
            device: 推理设备。
        """

        self.encoder = encoder.eval()
        self.device = device

        self._tokenize = clip.clip.tokenize

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        token: str | None = None,
        revision: str | None = None,
    ) -> "MobileCLIPTextEncoder":
        """从导出的模型目录中恢复文本编码器。

        Args:
            model_dir: 文本编码器模型目录，或可解析到 Hub 仓库的名称。
            device: 目标设备。
            token: 访问私有仓库时使用的令牌。
            revision: 需要解析的仓库版本。

        Returns:
            已完成 TorchScript 恢复并切换到 `eval` 模式的编码器实例。

        Raises:
            ValueError: 当配置声明的文本编码类型不受支持时抛出。
            FileNotFoundError: 当找不到内嵌文本编码器资产时抛出。
        """

        resolved_device = torch.device(device)
        config_kwargs: dict[str, object] = {}
        if token is not None:
            config_kwargs["token"] = token
        if revision is not None:
            config_kwargs["revision"] = revision
        resolved_config = AutoConfig.from_pretrained(str(model_dir), **config_kwargs)

        text_encoder_type = str(getattr(resolved_config, "text_encoder_type", ""))
        if text_encoder_type != "mobileclip2":
            raise ValueError(
                f"Unsupported text encoder type '{text_encoder_type}'. "
                "Only 'mobileclip2' is currently supported."
            )

        resolved_model_dir = Path(getattr(resolved_config, "name_or_path", "") or model_dir)
        asset_key = f"{ASSET_PREFIX}{getattr(resolved_config, 'text_encoder_asset')}"
        weights_path = resolve_weights_path(
            resolved_model_dir,
            token=token,
            revision=revision,
        )
        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find model.safetensors for text encoder model: {resolved_model_dir}"
            )

        _, asset_tensors = load_checkpoint_state(weights_path)
        available_assets = sorted(key.removeprefix(ASSET_PREFIX) for key in asset_tensors)
        asset_tensor = asset_tensors.get(asset_key)
        if asset_tensor is None:
            raise FileNotFoundError(
                f"Missing text encoder asset '{getattr(resolved_config, 'text_encoder_asset')}' "
                f"in text encoder model.safetensors. searched_dir={resolved_model_dir} "
                f"available_assets={available_assets or ['<none>']}"
            )

        asset_bytes = asset_tensor.detach().cpu().contiguous().numpy().tobytes()
        encoder = torch.jit.load(io.BytesIO(asset_bytes), map_location=resolved_device)
        return cls(
            encoder=encoder,
            device=resolved_device,
        )

    def tokenize(
        self,
        texts: Sequence[str],
        *,
        truncate: bool = True,
    ) -> torch.Tensor:
        """将文本列表编码为 CLIP token，并移动到目标设备。"""

        return self._tokenize(list(texts), truncate=truncate).to(self.device)

    @torch.inference_mode()
    def encode_tokens(
        self,
        tokens: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """对已分词的 token 张量执行前向编码。"""

        return self.encoder(tokens).to(dtype=dtype)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """对文本提示词执行完整编码流程。

        Args:
            texts: 原始提示词序列。
            dtype: 输出张量的数据类型。

        Returns:
            形状为 `[1, num_prompts, embedding_dim]` 的批量文本嵌入。

        Raises:
            ValueError: 当清洗后没有任何有效提示词时抛出。
        """

        prompts = [str(text).strip() for text in texts if str(text).strip()]
        if not prompts:
            raise ValueError("At least one non-empty text prompt is required.")

        tokens = self.tokenize(prompts)
        encoded = self.encode_tokens(tokens, dtype=dtype)
        return encoded.reshape(1, len(prompts), encoded.shape[-1])


YOLOETextEncoder = MobileCLIPTextEncoder
