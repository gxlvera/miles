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
"""Chat-template rendering used by the continuous token builders."""

import json
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
    messages = _template_ready(messages)
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


def render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """``apply_chat_template`` without tokenizing, for messages a processor has to encode with their images."""
    render_kwargs = dict(tokenize=False, tools=tools, **kwargs)
    messages = _template_ready(messages)
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, **render_kwargs)
    except Exception:
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        prefix = tokenizer.apply_chat_template(dummy_user, add_generation_prompt=False, **render_kwargs)
        full = tokenizer.apply_chat_template(
            dummy_user + messages, add_generation_prompt=add_generation_prompt, **render_kwargs
        )
        if not full.startswith(prefix):
            raise ValueError("chat template renders the dummy user turn differently inside a conversation") from None
        return full[len(prefix) :]


def _template_ready(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of ``messages`` in the shape HF Jinja templates expect.

    OpenAI messages carry tool-call ``arguments`` as a JSON string, but templates such as
    Qwen3.5's iterate them as a mapping (``arguments|items``). Assistant ``content: None`` becomes
    "". Only assistant messages with tool calls are copied.
    """
    ready = []
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            ready.append(message)
            continue
        tool_calls = []
        for tool_call in message["tool_calls"]:
            function = dict(tool_call.get("function") or {})
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pass
            tool_calls.append({**tool_call, "function": function})
        ready.append({**message, "content": message.get("content") or "", "tool_calls": tool_calls})
    return ready
