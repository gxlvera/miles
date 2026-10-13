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
"""Continuous token (CT) builders: the generic builder, Qwen and GLM.

The history is never re-rendered. Each newly appended message group is rendered inside a small
synthetic context (fake system + user, plus a fake assistant whose tool calls the tool results
answer), and the tokens the group adds are appended to the exact tokens the model already saw and
produced. A subclass then fixes the junction: Qwen, for example, stops generating at
``<|im_end|>`` while its template expects ``<|im_end|>\\n``.

A group with images is rendered to text and encoded by the model's processor together with its
images (``encode_images``), so the processor itself expands their placeholders. The synthetic
context never holds images.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Relative imports: this package is also copied into SGLang as ``sglang.srt.continuous_token``.
from .chat_template import apply_chat_template, render_chat_text
from .media import message_images

_APPEND_ROLES = frozenset({"tool", "user", "system", "assistant"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}


@dataclass(frozen=True)
class MergeResult:
    """The merged prompt.

    ``removed_prefix_token_count`` trailing tokens of the previous sequence were replaced; the agent
    server stores ``len(previous) - removed_prefix_token_count`` as the new turn's ``prefix_len``.
    """

    token_ids: list[int]
    removed_prefix_token_count: int = 0


class ContinuousTokenBuilder:
    """Build the first prompt of a conversation, or extend a token sequence with new messages.

    Callers use ``build_initial_tokens`` and ``merge_context_tokens``. A model family overrides
    ``_tokenize_tool_group`` / ``_tokenize_single_non_tool`` (how one appended group renders) and
    ``_merge_context_token_ids`` (the junction with the previous tokens). A builder given
    ``encode_images`` collects the processor outputs of the images it encodes in ``image_inputs``,
    so it serves one request.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        encode_images: Callable[[str, list[str]], tuple[list[int], dict[str, Any]]] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        # (text, image references) -> (token ids with the images expanded, processor outputs for the images)
        self.encode_images = encode_images
        self.image_inputs: list[dict[str, Any]] = []  # one entry per encode_images call, in prompt order

    def build_initial_tokens(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        """Render a conversation that has no reusable prefix (a new root)."""
        return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        """Append the messages after ``previous_messages`` to ``runtime_token_ids``, plus a generation prompt."""
        if updated_messages[: len(previous_messages)] != previous_messages:
            raise ValueError("Continuous Token messages must be append-only; prefix messages changed")
        appended_ids = self._tokenize_appended(updated_messages[len(previous_messages) :], tools=tools)
        return self._merge_context_token_ids(list(runtime_token_ids), appended_ids)

    def _tokenize_appended(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None) -> list[int]:
        groups = _append_groups(messages)
        token_ids: list[int] = []
        for index, group in enumerate(groups):
            role = group[0].get("role")
            if role not in _APPEND_ROLES:
                raise ValueError(f"Continuous Token can append roles {sorted(_APPEND_ROLES)}, got {role!r}")
            # The last group is rendered together with the generation prompt, so the history is never
            # rendered just to learn what add_generation_prompt=True adds.
            is_last = index == len(groups) - 1
            if role == "tool":
                token_ids += self._tokenize_tool_group(group, tools=tools, add_generation_prompt=is_last)
            elif role == "assistant" and is_last:
                raise ValueError("Continuous Token messages cannot end with an assistant message")
            else:
                token_ids += self._tokenize_single_non_tool(group[0], tools=tools, add_generation_prompt=is_last)
        return token_ids

    def _tokenize_tool_group(
        self, tool_messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, add_generation_prompt: bool
    ) -> list[int]:
        context = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, _assistant_calling(tool_messages)]
        return self._render_delta(context, tool_messages, add_generation_prompt=add_generation_prompt, tools=tools)

    def _tokenize_single_non_tool(
        self, message: dict[str, Any], *, tools: list[dict[str, Any]] | None, add_generation_prompt: bool
    ) -> list[int]:
        context = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        return self._render_delta(context, [message], add_generation_prompt=add_generation_prompt, tools=tools)

    def _merge_context_token_ids(self, previous_ids: list[int], appended_ids: list[int]) -> MergeResult:
        """Join the previous tokens and the appended tokens; subclasses fix the junction here."""
        return MergeResult(token_ids=previous_ids + appended_ids)

    def _render_delta(
        self,
        context: list[dict[str, Any]],
        appended: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        """Render the context, then the context plus ``appended``, and return the tokens ``appended`` adds."""
        context_ids = self._render_tokens(context, add_generation_prompt=False, tools=tools)
        full_ids = self._render_tokens(context + appended, add_generation_prompt=add_generation_prompt, tools=tools)
        if full_ids[: len(context_ids)] != context_ids:
            roles = [message.get("role") for message in appended]
            raise ValueError(f"Continuous Token token-id suffix diff failed for roles: {roles}")
        return full_ids[len(context_ids) :]

    def _render_tokens(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, tools: list[dict[str, Any]] | None
    ) -> list[int]:
        render_kwargs = dict(add_generation_prompt=add_generation_prompt, tools=tools, **self.chat_template_kwargs)
        images = message_images(messages)
        if not images:
            return apply_chat_template(self.tokenizer, messages, **render_kwargs)
        if self.encode_images is None:
            raise ValueError("messages contain images, but this Continuous Token builder has no image encoder")
        text = render_chat_text(self.tokenizer, messages, **render_kwargs)
        token_ids, image_inputs = self.encode_images(text, images)
        self.image_inputs.append(image_inputs)
        return token_ids


class QwenContinuousTokenBuilder(ContinuousTokenBuilder):
    """Qwen ChatML boundary handling.

    Qwen2, Qwen2.5, Qwen3 and Qwen3.5 templates render ``<|im_end|>\\n`` after a turn, while
    generation may stop at ``<|im_end|>``. When the previous tokens end there, insert the missing
    newline before appending context tokens.

    Note: when a Qwen3/Qwen3.5 context assistant message has no thinking, grouped CT encoding
    renders that assistant in a synthetic context and may produce an empty ``<think></think>``
    block. This only affects assistant messages CT has to render itself, i.e. ones the model did
    not generate; generated turns keep their original tokens.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected Qwen newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._im_end_id = require_token_id(tokenizer, "<|im_end|>")

    def _merge_context_token_ids(self, previous_ids: list[int], appended_ids: list[int]) -> MergeResult:
        if previous_ids and previous_ids[-1] == self._im_end_id:
            previous_ids = [*previous_ids, self._newline_id]
        return MergeResult(token_ids=previous_ids + appended_ids)


class GLMContinuousTokenBuilder(ContinuousTokenBuilder):
    """GLM-4.5/4.6/4.7 boundary handling.

    GLM stops generating at ``<|observation|>`` (a tool call) or ``<|user|>``, and the template opens
    the next tool result or user message with that same token. The appended group renders its own
    opening token, so the junction keeps exactly one:
    - the same token: drop it from the appended tokens, so the generated token stays as sampled;
    - a different one, e.g. the model called a tool but the next message is from the user: replace the
      generated token (``removed_prefix_token_count = 1``).
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._boundary_ids = frozenset(require_token_id(tokenizer, token) for token in ("<|observation|>", "<|user|>"))

    def _merge_context_token_ids(self, previous_ids: list[int], appended_ids: list[int]) -> MergeResult:
        if not previous_ids or previous_ids[-1] not in self._boundary_ids:
            return MergeResult(token_ids=previous_ids + appended_ids)
        if appended_ids and appended_ids[0] == previous_ids[-1]:
            return MergeResult(token_ids=previous_ids + appended_ids[1:])
        return MergeResult(token_ids=previous_ids[:-1] + appended_ids, removed_prefix_token_count=1)


def require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        if len(token_id) != 1:
            raise ValueError(f"Tokenizer returned multiple ids for required token {token!r}: {token_id!r}")
        token_id = token_id[0]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer does not define required token {token!r} (got {token_id!r})")
    return token_id


def _append_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive tool results form one group (they answer one assistant turn); any other message is a group alone."""
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "tool" and groups and groups[-1][0].get("role") == "tool":
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def _assistant_calling(tool_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """An assistant turn whose tool calls the given tool results answer, so the template renders them in place."""
    tool_calls = [
        {
            "id": f"continuous_token_call_{index}" if message.get("tool_call_id") is None else message["tool_call_id"],
            "type": "function",
            "function": {"name": message.get("name") or "continuous_token_tool", "arguments": {}},
        }
        for index, message in enumerate(tool_messages)
    ]
    return {"role": "assistant", "content": "", "reasoning_content": "reasoning", "tool_calls": tool_calls}
