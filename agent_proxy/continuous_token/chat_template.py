# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chat-template rendering used by the continuous token builders.

Copied from verl (``verl/utils/tokenizer/chat_template.py`` and ``tokenizer.py``, branch
``gxl-ct-rollback-support``) so that the proxy never imports verl. Text path only:
``tokenize=True`` and a plain token-id list as the result.
"""

from typing import Any


def normalize_token_ids(tokenized_output: Any) -> list[int]:
    """Flatten what ``apply_chat_template(tokenize=True)`` returns into ``list[int]``.

    Transformers 4 returns a list, Transformers 5 may return a ``BatchEncoding`` or a
    batch of one.
    """
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        token_ids = tokenized_output.get("input_ids", tokenized_output)
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    token_ids = list(token_ids)
    if len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])
    return [int(token_id) for token_id in token_ids]


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[int]:
    """Render and tokenize ``messages``.

    Qwen3.5's template raises when there is no user message, which happens for the
    synthetic contexts CT renders (e.g. system + assistant). In that case render behind
    a dummy user turn and cut the dummy's tokens off again.
    """
    render_kwargs = dict(tokenize=True, tools=tools, return_dict=False, **kwargs)
    try:
        return normalize_token_ids(
            tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, **render_kwargs)
        )
    except Exception:
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        prefix = normalize_token_ids(
            tokenizer.apply_chat_template(dummy_user, add_generation_prompt=False, **render_kwargs)
        )
        full = normalize_token_ids(
            tokenizer.apply_chat_template(
                dummy_user + messages, add_generation_prompt=add_generation_prompt, **render_kwargs
            )
        )
        return full[len(prefix) :]
