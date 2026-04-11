from __future__ import annotations

import posixpath
from pathlib import Path

from safetensors.torch import load_file
from transformers.utils.hub import cached_file

ASSET_PREFIX = "__asset__."


def resolve_model_file(
    name_or_path: str | Path,
    filename: str,
    **cached_file_kwargs,
) -> Path | None:
    """解析模型目录中的指定文件。

    Args:
        name_or_path: 本地模型目录，或 Hugging Face Hub 仓库名。
        filename: 目标文件名。
        **cached_file_kwargs: 透传给 `transformers.utils.hub.cached_file` 的参数。

    Returns:
        解析成功时返回文件路径；找不到时返回 `None`。
    """

    candidate = Path(name_or_path) / filename
    if candidate.exists():
        return candidate

    try:
        resolved = cached_file(
            str(name_or_path),
            filename,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **cached_file_kwargs,
        )
    except Exception:
        resolved = None

    return Path(resolved) if resolved else None


def resolve_weights_path(name_or_path: str | Path, **cached_file_kwargs) -> Path | None:
    """定位 `model.safetensors` 权重文件。"""

    return resolve_model_file(name_or_path, "model.safetensors", **cached_file_kwargs)


def resolve_text_encoder_dir(
    model_dir: str | Path,
    *,
    text_encoder_path: str | Path | None = None,
) -> str | Path:
    """解析文本编码器目录。

    当配置没有显式指定子目录时，默认复用模型目录；若提供了相对路径，
    会优先拼接到本地模型目录下，方便本地导出产物直接携带文本编码器资源。
    """

    base_dir = Path(model_dir)
    if text_encoder_path is None:
        return base_dir if base_dir.exists() else str(model_dir)

    candidate = Path(text_encoder_path)
    if candidate.is_absolute():
        return candidate
    if base_dir.exists():
        return (base_dir / candidate).resolve()
    return posixpath.normpath(posixpath.join(str(model_dir), str(text_encoder_path)))


def load_checkpoint_state(
    weights_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """读取 safetensors 并拆分普通权重与资产张量。

    Args:
        weights_path: `model.safetensors` 文件路径。

    Returns:
        一个二元组：
        - 第一个字典为可直接加载到 `nn.Module` 的普通参数。
        - 第二个字典为带 `ASSET_PREFIX` 前缀的附加资产。
    """

    raw_tensors = load_file(str(weights_path))
    # 约定使用固定前缀标记非参数类资产，便于与 state_dict 分离。
    state_dict = {
        key: value
        for key, value in raw_tensors.items()
        if not key.startswith(ASSET_PREFIX)
    }
    asset_tensors = {
        key: value
        for key, value in raw_tensors.items()
        if key.startswith(ASSET_PREFIX)
    }
    return state_dict, asset_tensors
