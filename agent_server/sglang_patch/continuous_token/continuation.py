"""Extend a conversation's tokens with its new messages: what SGLang runs for agent_server requests.

The agent server sends the parent turn's full token sequence and the request messages, of which the
first ``prefix_message_count`` are covered by those tokens. Continuous token renders only the
messages after them. Groups with images are encoded by the model's processor, which expands their
placeholders. The tokens before ``prefix_len`` are never touched and never decoded.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .builder import ContinuousTokenBuilder, GLMContinuousTokenBuilder, QwenContinuousTokenBuilder
from .media import load_image, message_images

# Root HF ``config.json`` model_type values whose templates end a turn with ``<|im_end|>\n``.
_QWEN_MODEL_TYPES = frozenset(
    {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
)
# GLM-4.5/4.6/4.7 (glm4_moe) and GLM-4.7-Flash (glm4_moe_lite): generation stops at the next message's opening token.
_GLM_MODEL_TYPES = frozenset({"glm4_moe", "glm4_moe_lite"})
_BUILDERS_BY_MODEL_TYPE: dict[str, type[ContinuousTokenBuilder]] = {
    **dict.fromkeys(_QWEN_MODEL_TYPES, QwenContinuousTokenBuilder),
    **dict.fromkeys(_GLM_MODEL_TYPES, GLMContinuousTokenBuilder),
}
# Processor outputs that are text, not image inputs.
_TEXT_KEYS = frozenset({"input_ids", "attention_mask", "mm_token_type_ids"})


@dataclass(frozen=True)
class Continuation:
    input_ids: list[int]  # the whole prompt, every image placeholder expanded
    prefix_len: int  # tokens of the sent ids kept in front of the new ones
    # Processor outputs for every image of the prompt in order (e.g. pixel_values, image_grid_thw), or None.
    image_inputs: dict[str, torch.Tensor] | None


def create_builder(
    tokenizer: Any,
    *,
    model_type: str | None,
    chat_template_kwargs: dict[str, Any] | None = None,
    processor: Any = None,
) -> ContinuousTokenBuilder:
    """A builder for one request. With a processor, groups with images are encoded together with their images."""
    builder_cls = _BUILDERS_BY_MODEL_TYPE.get(model_type, ContinuousTokenBuilder)
    encode_images = None if processor is None else lambda text, refs: _encode_with_images(processor, text, refs)
    return builder_cls(tokenizer, chat_template_kwargs=chat_template_kwargs, encode_images=encode_images)


def continue_prompt(
    tokenizer: Any,
    processor: Any,
    *,
    model_type: str | None,
    input_ids: list[int],
    messages: list[dict[str, Any]],
    prefix_message_count: int,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Continuation:
    """``input_ids`` are the tokens of ``messages[:prefix_message_count]`` (empty for a new conversation)."""
    if processor is None and message_images(messages):
        raise ValueError("the served model takes no images")
    builder = create_builder(
        tokenizer, model_type=model_type, chat_template_kwargs=chat_template_kwargs, processor=processor
    )
    if prefix_message_count == 0:
        if input_ids:
            raise ValueError("a request with prefix_message_count=0 starts a conversation and sends no input_ids")
        prompt, prefix_len = builder.build_initial_tokens(messages, tools=tools), 0
    else:
        merge = builder.merge_context_tokens(messages[:prefix_message_count], messages, input_ids, tools=tools)
        prompt, prefix_len = merge.token_ids, len(input_ids) - merge.removed_prefix_token_count
        if prompt[:prefix_len] != input_ids[:prefix_len]:
            raise ValueError("continuous token changed tokens before its declared merge point")

    # Earlier images are already expanded in input_ids; only their pixels are needed again.
    history = message_images(messages[:prefix_message_count])
    parts = builder.image_inputs
    if history:
        parts = [_process_images(processor, history), *parts]
    image_inputs = {key: torch.cat([part[key] for part in parts]) for key in parts[0]} if parts else None
    if image_inputs is not None:
        _check_image_runs(prompt, processor.image_token_id, num_images=len(message_images(messages)))
    return Continuation(input_ids=prompt, prefix_len=prefix_len, image_inputs=image_inputs)


def _encode_with_images(processor: Any, text: str, refs: list[str]) -> tuple[list[int], dict[str, Any]]:
    output = processor(text=[text], images=[load_image(ref) for ref in refs], return_tensors="pt")
    return output["input_ids"][0].tolist(), {key: value for key, value in output.items() if key not in _TEXT_KEYS}


def _process_images(processor: Any, refs: list[str]) -> dict[str, Any]:
    output = processor.image_processor(images=[load_image(ref) for ref in refs], return_tensors="pt")
    return dict(output)


def _check_image_runs(token_ids: list[int], image_token_id: int, *, num_images: int) -> None:
    """Each image must own exactly one run of placeholder tokens; SGLang assigns runs to images in order."""
    runs = sum(
        token == image_token_id and (index == 0 or token_ids[index - 1] != image_token_id)
        for index, token in enumerate(token_ids)
    )
    if runs != num_images:
        raise ValueError(f"the prompt has {runs} image placeholder runs for {num_images} images")
