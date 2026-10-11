"""Continuous token builders, copied from verl so the proxy does not depend on it (see builder.py)."""

import logging
from typing import Any

from agent_proxy.continuous_token.builder import ContinuousTokenBuilder, MergeResult, QwenContinuousTokenBuilder

logger = logging.getLogger(__name__)

# Root HF ``config.json`` model_type values whose templates end a turn with ``<|im_end|>\n``.
_QWEN_MODEL_TYPES = frozenset({"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"})


def create_builder(
    tokenizer: Any,
    *,
    model_type: str | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> ContinuousTokenBuilder:
    if model_type in _QWEN_MODEL_TYPES:
        return QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs=chat_template_kwargs)
    logger.warning(
        "No model-specific continuous token builder for model_type=%r; using the generic builder, which adds "
        "no boundary tokens between a stopped generation and the next message",
        model_type,
    )
    return ContinuousTokenBuilder(tokenizer, chat_template_kwargs=chat_template_kwargs)


__all__ = ["ContinuousTokenBuilder", "MergeResult", "QwenContinuousTokenBuilder", "create_builder"]
