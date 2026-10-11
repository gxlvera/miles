"""Data passed between the Router, a replica, its trajectory store and the inference backend."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from miles.utils.types import WeightVersionsPerCall


class ProxyError(Exception):
    """An error the replica returns to the client with this HTTP status. Nothing is committed."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class SessionHandle:
    """What the Router returns for a new session. After this, nothing goes through the Router."""

    session_id: str
    replica_url: str  # owner of the session: collect and delete go here
    agent_base_url: str  # the agent appends /v1/chat/completions or /v1/messages


@dataclass(frozen=True)
class R3Chunk:
    """Routed-expert rows (routing replay) returned for one generation.

    Row ``r`` belongs to token position ``r`` of the turn's full sequence, and a sequence of
    ``n`` tokens has ``n - 1`` rows. ``values`` covers rows ``[row_start, row_start + len(values))``;
    rows before ``row_start`` are the parent's, which is why
    ``r3(node) = r3(parent)[:row_start] + values``. A backend that recomputes every row returns
    ``row_start == 0`` and nothing is inherited.
    """

    row_start: int
    values: np.ndarray  # [num_rows, num_layers, topk], dtype as decoded from the backend (int32)


@dataclass(frozen=True)
class PreparedTurn:
    """A request after prepare: its parent is fixed and its prompt is rendered.

    Lives only while the request is in flight; the full ``input_ids`` are never stored.
    """

    parent_id: int | None
    prefix_len: int  # tokens of the parent's sequence kept in front of this prompt
    input_ids: list[int]  # the complete prompt sent to the backend
    new_messages: list[dict[str, Any]]  # request messages after the parent's path
    context_key: bytes  # tools + template kwargs; prefixes are shared only within one key
    fingerprints: list[bytes]  # one per request message, reused to index the turn on commit
    reusable_r3_rows: int  # parent routing rows still valid for this prompt


@dataclass(frozen=True)
class Generation:
    """The parts of one backend reply the proxy keeps or returns."""

    response: dict[str, Any]  # the full OpenAI reply, meta_info included
    message: dict[str, Any]  # the parsed assistant message, as the agent will replay it
    output_tokens: list[int]
    output_logprobs: list[float]
    finish_reason: str
    meta_info: dict[str, Any]
    r3: R3Chunk | None

    @property
    def response_id(self) -> str:
        return self.response.get("id") or ""


@dataclass(frozen=True)
class TurnNode:
    """One committed generation. Holds only what this turn added to its parent's sequence:

        tokens(node) = tokens(parent)[:prefix_len] + input_delta + output_tokens

    ``prefix_len`` is normally the parent's full length. It is shorter only when continuous
    token replaced the tail of the parent's sequence; the parent itself is never modified,
    since other branches may share it. Lists are never mutated after commit.
    """

    node_id: int  # also the commit order within the session
    parent_id: int | None
    prefix_len: int
    input_delta: list[int]  # new prompt tokens: tool results, user messages, template tokens
    output_tokens: list[int]
    output_logprobs: list[float]
    delta_messages: list[dict[str, Any]]  # new request messages plus the assistant reply
    finish_reason: str
    weight_versions: WeightVersionsPerCall  # absolute positions, valid on every path through this node
    response_id: str
    committed_at: float
    r3: R3Chunk | None = None

    @property
    def prompt_len(self) -> int:
        return self.prefix_len + len(self.input_delta)

    @property
    def end(self) -> int:
        return self.prompt_len + len(self.output_tokens)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"
