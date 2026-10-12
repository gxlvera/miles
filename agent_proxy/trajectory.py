"""Per-session trajectory store: incremental turn nodes plus a message trie for finding the parent.

One node per generation. Retries and rollbacks become branches, and requests that share no
prefix with earlier turns (compaction, sub-agents) become extra roots. A node keeps only its own
delta, so every token, logprob and message is stored once; a full sequence exists only
transiently, as the prompt of an in-flight request and while exporting samples.

Finding the parent does not scan branches. Each request message is reduced to a fingerprint
and the request walks one trie edge per message; trie positions where a committed turn ends
record that turn's id.
"""

import hashlib
import itertools
import json
import time
from dataclasses import replace
from typing import Any

import numpy as np

from agent_proxy.continuous_token import ContinuousTokenBuilder
from agent_proxy.protocol import Generation, PreparedTurn, TurnNode
from miles.utils.types import Sample, WeightVersionsPerCall

_STATUS_BY_FINISH_REASON = {"length": Sample.Status.TRUNCATED, "abort": Sample.Status.ABORTED}


def message_fingerprint(message: dict[str, Any]) -> bytes:
    """Digest of what a message means to the model; equal digests are treated as the same history step.

    The projection keeps role, content, reasoning_content and tool_calls, with empty values
    collapsed to None. Two fields are relaxed because they change on a client round trip without
    changing what the model saw:
    - a tool call keeps only its function name and arguments, with JSON arguments compared as
      parsed objects (``id``/``index``/``type`` and argument whitespace or key order differ after
      e.g. an Anthropic <-> OpenAI conversion);
    - content made only of text blocks counts as the joined string.
    The stored tokens of a matched turn are reused as they are, so ignoring these fields cannot
    change any prompt token.
    """
    tool_calls = [
        {"name": (call.get("function") or {}).get("name"), "arguments": _canonical_arguments(call)}
        for call in message.get("tool_calls") or []
    ]
    return _digest(
        {
            "role": message.get("role"),
            "content": _canonical_content(message.get("content")),
            "reasoning_content": message.get("reasoning_content") or None,
            "tool_calls": tool_calls or None,
        }
    )


def context_key(tools: list[dict[str, Any]] | None, template_kwargs: dict[str, Any]) -> bytes:
    """Prompts rendered with different tools or template kwargs never share a prefix."""
    return _digest({"tools": tools or None, "template_kwargs": template_kwargs})


class _TrieNode:
    __slots__ = ("children", "turn_ids")

    def __init__(self) -> None:
        self.children: dict[bytes, _TrieNode] = {}
        self.turn_ids: list[int] = []  # committed turns whose full message path ends here, in commit order


class TrajectoryManager:
    """All turns of one session.

    ``prepare`` and ``commit`` are synchronous and a replica serves all sessions from one event
    loop, so each call runs atomically with respect to the session's other requests and no lock
    is needed. The parent is fixed in ``prepare`` and never revisited: two requests prepared from
    the same turn become siblings, whichever finishes first.
    """

    def __init__(self, *, message_normalizer=None) -> None:
        self.nodes: list[TurnNode] = []
        self._tries: dict[bytes, _TrieNode] = {}
        self._message_normalizer = message_normalizer

    def _fingerprint(self, message: dict[str, Any]) -> bytes:
        if self._message_normalizer is not None:
            message = self._message_normalizer(message)
        return message_fingerprint(message)

    def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        builder: ContinuousTokenBuilder,
        context_key: bytes,
    ) -> PreparedTurn:
        """Pick the parent turn and render the prompt with continuous token."""
        fingerprints = [self._fingerprint(message) for message in messages]
        parent_id, depth = self._find_parent(context_key, fingerprints)

        if parent_id is None:
            input_ids = builder.build_initial_tokens(messages, tools=tools)
            return PreparedTurn(
                parent_id=None,
                prefix_len=0,
                input_ids=input_ids,
                new_messages=list(messages),
                context_key=context_key,
                fingerprints=fingerprints,
                reusable_r3_rows=0,
            )

        parent_tokens = self.tokens(parent_id)
        # CT checks that ``updated`` extends ``previous``. The request's own first ``depth`` messages
        # matched the parent's path, so pass those instead of reassembling the stored messages.
        merge = builder.merge_context_tokens(messages[:depth], messages, parent_tokens, tools=tools)
        prefix_len = len(parent_tokens) - merge.removed_prefix_token_count
        # Delta storage is only correct if CT kept the parent's tokens as they were.
        if merge.token_ids[:prefix_len] != parent_tokens[:prefix_len]:
            raise ValueError("continuous token changed tokens before its declared merge point")
        return PreparedTurn(
            parent_id=parent_id,
            prefix_len=prefix_len,
            input_ids=merge.token_ids,
            new_messages=messages[depth:],
            context_key=context_key,
            fingerprints=fingerprints,
            # Row r depends on tokens [0, r], so rows stay valid up to the first changed token.
            reusable_r3_rows=min(prefix_len, len(parent_tokens) - 1),
        )

    def commit(self, prepared: PreparedTurn, generation: Generation) -> TurnNode:
        prompt_len = len(prepared.input_ids)
        node = TurnNode(
            node_id=len(self.nodes),
            parent_id=prepared.parent_id,
            prefix_len=prepared.prefix_len,
            input_delta=prepared.input_ids[prepared.prefix_len :],
            output_tokens=generation.output_tokens,
            output_logprobs=generation.output_logprobs,
            delta_messages=[*prepared.new_messages, generation.message],
            finish_reason=generation.finish_reason,
            weight_versions=WeightVersionsPerCall.from_meta_info(
                generation.meta_info, output_end=prompt_len + len(generation.output_tokens)
            ),
            response_id=generation.response_id,
            committed_at=time.time(),
            r3=generation.r3,
        )
        self.nodes.append(node)
        # A generation cut by max_tokens has no closing <|im_end|>; continuing after it would train on a
        # layout the template never produces. Keeping it out of the trie makes a replay of it attach to an
        # earlier turn, where CT renders the cut text as ordinary context.
        if not node.truncated:
            self._index(
                prepared.context_key, [*prepared.fingerprints, self._fingerprint(generation.message)], node.node_id
            )
        return node

    def path(self, node_id: int) -> list[TurnNode]:
        """Nodes from the root down to ``node_id``."""
        path = []
        current: int | None = node_id
        while current is not None:
            node = self.nodes[current]
            path.append(node)
            current = node.parent_id
        path.reverse()
        return path

    def tokens(self, node_id: int) -> list[int]:
        """Rebuild one turn's full sequence (prompt + output) from the deltas on its path."""
        tokens: list[int] = []
        for node in self.path(node_id):
            del tokens[node.prefix_len :]
            tokens += node.input_delta
            tokens += node.output_tokens
        return tokens

    def leaf_samples(self, *, max_seq_len: int | None, tokenizer: Any) -> list[Sample]:
        """One raw training sample per root-to-leaf path, in commit order.

        These are the input of the picker and postprocessor hooks, which read each sample's
        ``leaf`` and ``accumulated_token_ids`` metadata.
        """
        samples = []
        for leaf in self._leaves():
            full_path = self.path(leaf.node_id)
            path = full_path
            if max_seq_len is not None:
                # Cut at turn level: a turn whose prompt already fills the budget is dropped together with
                # everything after it, and a turn whose output crosses the budget is cut and marked TRUNCATED.
                path = list(itertools.takewhile(lambda node: node.prompt_len < max_seq_len, full_path))
                if not path:
                    continue
            sample = _assemble_sample(path, tokenizer)
            sample.metadata = {
                "accumulated_token_ids": sample.tokens,
                "leaf": {
                    "node_id": leaf.node_id,
                    "parent": leaf.parent_id,
                    "path_node_ids": [node.node_id for node in full_path],
                    "response_id": leaf.response_id,
                },
            }
            if max_seq_len is not None and len(sample.tokens) > max_seq_len:
                sample.strip_last_output_tokens(len(sample.tokens) - max_seq_len, tokenizer)
                sample.status = Sample.Status.TRUNCATED
            samples.append(sample)
        return samples

    def tree_metadata(self) -> dict[str, Any]:
        """Node and leaf tables read by the picker and postprocessor hooks."""
        nodes = [
            {
                "id": node.node_id,
                "parent": node.parent_id,
                "seq": node.node_id,
                "truncated": node.truncated,
                "committed_at": node.committed_at,
                "completion_span": [node.prompt_len, node.end],
                "num_tokens": node.end,
                "response_id": node.response_id,
            }
            for node in self.nodes
        ]
        leaves = [
            {"node_id": leaf.node_id, "path_node_ids": [node.node_id for node in self.path(leaf.node_id)]}
            for leaf in self._leaves()
        ]
        return {"nodes": nodes, "leaves": leaves}

    def clear(self) -> None:
        self.nodes.clear()
        self._tries.clear()

    def _find_parent(self, context_key: bytes, fingerprints: list[bytes]) -> tuple[int | None, int]:
        """Deepest committed turn whose message path is a proper prefix of the request.

        Returns the turn id (None for a new root) and how many request messages its path covers.
        The last request message is never consumed, because the new turn must append at least one
        message after its parent. Several turns can end at the same trie position when independent
        generations produced the same text; their tokens or logprobs may still differ, so they stay
        separate nodes and the latest commit is used.
        """
        trie = self._tries.get(context_key)
        parent_id, depth = None, 0
        for index, fingerprint in enumerate(fingerprints[:-1]):
            if trie is None:
                break
            trie = trie.children.get(fingerprint)
            if trie is not None and trie.turn_ids:
                parent_id, depth = trie.turn_ids[-1], index + 1
        return parent_id, depth

    def _index(self, context_key: bytes, fingerprints: list[bytes], node_id: int) -> None:
        trie = self._tries.setdefault(context_key, _TrieNode())
        for fingerprint in fingerprints:
            trie = trie.children.setdefault(fingerprint, _TrieNode())
        trie.turn_ids.append(node_id)

    def _leaves(self) -> list[TurnNode]:
        parents = {node.parent_id for node in self.nodes}
        return [node for node in self.nodes if node.node_id not in parents]


def _assemble_sample(path: list[TurnNode], tokenizer: Any) -> Sample:
    """Replay one root-to-leaf path into a Miles Sample with a single growing buffer per field.

    Prompt-side tokens (tool results, user messages, template and CT boundary tokens) get loss
    mask 0 and logprob 0; model output gets mask 1 and its sampled logprob. The response region
    starts at the first model output.
    """
    tokens: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    weight_versions: list[WeightVersionsPerCall] = []
    for node in path:
        if node.prefix_len < len(tokens):
            for buffer in (tokens, mask, logprobs):
                del buffer[node.prefix_len :]
            weight_versions = [_clip_spans(call, end=node.prefix_len) for call in weight_versions]
        tokens += node.input_delta
        mask += [0] * len(node.input_delta)
        logprobs += [0.0] * len(node.input_delta)
        tokens += node.output_tokens
        mask += [1] * len(node.output_tokens)
        logprobs += node.output_logprobs
        # A copy: Sample.strip_last_output_tokens edits spans in place, and nodes are shared by samples.
        weight_versions.append(_clip_spans(node.weight_versions, end=node.end))

    response_start = path[0].prompt_len
    return Sample(
        tokens=tokens,
        response=tokenizer.decode(tokens[response_start:]),
        response_length=len(tokens) - response_start,
        loss_mask=mask[response_start:],
        rollout_log_probs=logprobs[response_start:],
        weight_versions=weight_versions,
        rollout_routed_experts=_routed_experts(path, num_rows=len(tokens) - 1),
        status=_STATUS_BY_FINISH_REASON.get(path[-1].finish_reason, Sample.Status.COMPLETED),
    )


def _routed_experts(path: list[TurnNode], *, num_rows: int) -> np.ndarray | None:
    """Rebuild the routing-replay rows of a path from the per-turn chunks.

    Walks up from the leaf and takes from each chunk only the rows no descendant replaced, then
    concatenates once. A gap means the backend skipped rows the proxy asked it to keep, and the
    sample cannot be replayed, so it is an error rather than zero padding.
    """
    if path[-1].r3 is None:
        return None
    pieces = []
    needed = num_rows
    for node in reversed(path):
        if needed == 0:
            break
        chunk = node.r3
        if chunk is None:
            raise ValueError(f"turn {node.node_id} has no routing replay rows but a descendant needs them")
        take = needed - chunk.row_start
        if take <= 0:
            continue
        if len(chunk.values) < take:
            raise ValueError(f"turn {node.node_id} returned {len(chunk.values)} routing rows, {take} are needed")
        pieces.append(chunk.values[:take])
        needed = chunk.row_start
    if needed:
        raise ValueError(f"routing replay rows [0, {needed}) are missing")
    return np.concatenate(pieces[::-1])


def _clip_spans(call: WeightVersionsPerCall, *, end: int) -> WeightVersionsPerCall:
    return WeightVersionsPerCall(
        spans=[replace(span, abs_end=min(span.abs_end, end)) for span in call.spans if span.abs_start < end]
    )


def _canonical_content(content: Any) -> Any:
    if isinstance(content, list) and all(isinstance(block, dict) and block.get("type") == "text" for block in content):
        content = "".join(block.get("text", "") for block in content)
    return content or None


def _canonical_arguments(tool_call: dict[str, Any]) -> Any:
    arguments = (tool_call.get("function") or {}).get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _digest(value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.blake2b(encoded.encode(), digest_size=16).digest()
