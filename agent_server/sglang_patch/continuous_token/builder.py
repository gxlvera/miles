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
"""Continuous token (CT) builders: the generic builder and Qwen.

Ported from verl ``verl/utils/tokenizer/continuous_token.py``, branch ``gxl-ct-rollback-support``
(commit e18c3167).

The builder only builds prompts. The backend's output tokens are appended as they are, and loss
masks are derived from the turn tree when samples are exported. A rollback is a new branch from
an earlier turn, which is an ordinary context merge.

How CT builds a prompt: the history is never re-rendered. Each newly appended message group is
rendered inside a small synthetic context (fake system + user, plus a fake assistant with tool
calls for tool results), and the tokens that the group adds are appended to the exact tokens the
model already saw and produced. A subclass then fixes the junction; Qwen, for example, stops
generating at ``<|im_end|>`` while its template expects ``<|im_end|>\n``.

Images: a group with images is rendered to text and encoded by the model's processor together with
its images (``encode_images``), so its placeholders come out expanded by the processor itself. The
synthetic context never holds images.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Relative imports: this package is also copied into SGLang as ``sglang.srt.continuous_token``.
from .chat_template import apply_chat_template, render_chat_text
from .media import message_images

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system", "assistant"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT = "reasoning"
_DUMMY_TOOL_NAME = "continuous_token_tool"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    """The merged prompt plus how it changed at the junction with the previous tokens.

    ``inserted_token_ids`` are boundary tokens CT added itself (never model output), and
    ``removed_prefix_token_count`` counts trailing tokens of the previous sequence that were
    dropped before appending. The agent server stores ``len(previous) - removed_prefix_token_count``
    as the new turn's ``prefix_len``.
    """

    token_ids: list[int]
    appended_token_count: int
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class ContinuousTokenBuilder:
    """Build the first prompt of a conversation and extend an existing token sequence with new messages.

    Public API: ``build_initial_tokens`` and ``merge_context_tokens``. A builder given ``encode_images``
    collects the processor outputs of the images it encoded in ``image_inputs``, so it serves one request.
    Template-specific behavior lives in the hooks ``_tokenize_tool_group``,
    ``_tokenize_single_non_tool``, ``_merge_context_group`` and ``_merge_context_token_ids``.
    """

    allowed_append_roles: frozenset[str] = _SUPPORTED_APPEND_ROLES

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
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
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
        """Append the messages after ``previous_messages`` to ``runtime_token_ids``, ending with a generation prompt."""
        appended_messages = updated_messages[len(previous_messages) :]
        appended_ids = self.tokenize_context_incremental_messages(previous_messages, updated_messages, tools=tools)
        return self._merge_context_token_ids(
            runtime_token_ids,
            appended_ids,
            previous_messages=previous_messages,
            appended_messages=appended_messages,
        )

    def tokenize_context_incremental_messages(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages) :]
        if not appended_messages:
            return []
        incremental_ids: list[int] = []
        processed_messages = list(previous_messages)

        groups = self._iter_append_groups(appended_messages)
        # Render the generation prompt together with the last group, so the history is never
        # rendered just to learn what ``add_generation_prompt=True`` adds.
        fuse_generation_prompt = self._should_fuse_generation_prompt_with_last_group()

        for index, group in enumerate(groups):
            add_generation_prompt = fuse_generation_prompt and index == len(groups) - 1
            role = group[0].get("role")
            if role == "tool":
                group_token_ids = self._tokenize_tool_group(
                    group,
                    previous_messages=processed_messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            elif role in {"user", "system", "assistant"}:
                # System appends can represent retry/control messages; unsupported templates fail in the suffix diff.
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                if role == "assistant" and index == len(groups) - 1:
                    raise ValueError("Continuous Token context incremental messages cannot end with assistant")
                group_token_ids = self._tokenize_single_non_tool(
                    group[0],
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            self._merge_context_group(
                incremental_ids,
                group_token_ids,
                group=group,
                processed_messages=processed_messages,
                tools=tools,
            )
            processed_messages.extend(group)

        if not fuse_generation_prompt:
            incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def render_delta_token_id(
        self,
        prefix_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render prefix and prefix + appended, and return the token suffix the appended messages add."""
        if message_images(prefix_messages):
            # The processor would encode the prefix's images too and the diff would cut through them.
            raise ValueError("Continuous Token renders images only in appended messages, after an image-free prefix")
        prefix_token_ids = self._render_tokens(prefix_messages, add_generation_prompt=False, tools=tools)
        full_token_ids = self._render_tokens(
            prefix_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if full_token_ids[: len(prefix_token_ids)] != prefix_token_ids:
            roles = [message.get("role") for message in appended_messages] or ["generation_prompt"]
            raise ValueError(f"Continuous Token token-id suffix diff failed for roles: {roles}")
        return full_token_ids[len(prefix_token_ids) :]

    def _merge_context_group(
        self,
        incremental_ids: list[int],
        group_token_ids: list[int],
        *,
        group: list[dict[str, Any]],
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Merge one encoded context group into the incremental token stream."""
        incremental_ids.extend(group_token_ids)

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        """Join the previous tokens and the appended tokens; subclasses fix the junction here."""
        return MergeResult(
            token_ids=list(runtime_token_ids) + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
        )

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        images = message_images(messages)
        if images:
            if self.encode_images is None:
                raise ValueError("messages contain images, but this Continuous Token builder has no image encoder")
            text = render_chat_text(
                self.tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                **self.chat_template_kwargs,
            )
            token_ids, image_inputs = self.encode_images(text, images)
            self.image_inputs.append(image_inputs)
            return token_ids
        return apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        synthetic_assistant = self._synthetic_assistant_for_tools(tool_messages)
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            tool_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            [message],
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        """Whether the last append group's render also adds the generation prompt.

        A builder whose template derives the generation scaffold from more than the appended
        group returns False and implements ``_tokenize_generation_prompt_delta`` instead.
        """
        return True

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Tokens added only by ``add_generation_prompt=True``."""
        return self.render_delta_token_id(updated_messages, [], add_generation_prompt=True, tools=tools)

    def _iter_append_groups(self, appended_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Consecutive tool results form one group (they answer one assistant turn); every other message is its own group."""
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(appended_messages):
            end = index + 1
            if appended_messages[index].get("role") == "tool":
                while end < len(appended_messages) and appended_messages[end].get("role") == "tool":
                    end += 1
            groups.append(appended_messages[index:end])
            index = end
        return groups

    def _assert_append_only(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
    ) -> None:
        if len(updated_messages) < len(previous_messages):
            raise ValueError("Continuous Token messages must be append-only; updated_messages is shorter")
        if updated_messages[: len(previous_messages)] != previous_messages:
            raise ValueError("Continuous Token messages must be append-only; prefix messages changed")
        for message in updated_messages[len(previous_messages) :]:
            role = message.get("role")
            if role not in self.allowed_append_roles:
                raise ValueError(
                    f"Continuous Token only supports appending roles {sorted(self.allowed_append_roles)}, got {role!r}"
                )

    def _synthetic_assistant_for_tools(self, tool_messages: list[dict[str, Any]]) -> dict[str, Any]:
        """An assistant turn whose tool calls the given tool results answer, so the template renders them in place."""
        tool_calls = [
            {
                "id": _tool_call_id_or_dummy(tool_message, index),
                "type": "function",
                "function": {"name": _tool_message_name_or_dummy(tool_message), "arguments": {}},
            }
            for index, tool_message in enumerate(tool_messages)
        ]
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }


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

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            inserted_token_ids=inserted_token_ids,
        )


def require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        if len(token_id) != 1:
            raise ValueError(f"Tokenizer returned multiple ids for required token {token!r}: {token_id!r}")
        token_id = token_id[0]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer does not define required token {token!r} (got {token_id!r})")
    return token_id


def _tool_message_name_or_dummy(tool_message: dict[str, Any]) -> str:
    if tool_message.get("name"):
        return str(tool_message["name"])
    return _DUMMY_TOOL_NAME


def _tool_call_id_or_dummy(tool_message: dict[str, Any], index: int) -> Any:
    if tool_message.get("tool_call_id") is not None:
        return tool_message["tool_call_id"]
    return f"continuous_token_call_{index}"
