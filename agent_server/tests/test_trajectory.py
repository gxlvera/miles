"""TrajectoryManager on a toy tokenizer: parent selection, delta storage and sample export.

SGLang renders prompts (continuous token runs there); ``_render_like_sglang`` plays its part. Every
committed turn is also recorded the flat way (full prompt + output), and the delta store must
rebuild exactly those sequences.
"""

import numpy as np
import pytest

from agent_server.protocol import Generation, PromptEdit, R3Chunk
from agent_server.trajectory import TrajectoryManager, context_key, message_fingerprint
from miles.rollout.session.v2.picker_hub import drop_same_prompt_retries
from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.utils.types import Sample

END = 2  # plays <|im_end|>
GEN = 7  # plays the generation prompt
_ROLE_IDS = {"system": 101, "user": 102, "assistant": 103, "tool": 104}
KEY = context_key(None, {})

SYSTEM = {"role": "system", "content": "sys"}
USER = {"role": "user", "content": "go"}


def _render(messages):
    ids = []
    for message in messages:
        ids += [_ROLE_IDS[message["role"]], *(ord(c) for c in message.get("content") or ""), END]
    return ids


def _render_like_sglang(prepared, *, replace_tail):
    """Each message after the parent's path rendered on its own, plus the generation prompt.

    With ``replace_tail`` it also replaces the parent's final END, as a builder fixing the junction would.
    """
    appended = _render(prepared.new_messages) + [GEN]
    removed = 1 if replace_tail and prepared.parent_tokens else 0
    delta = [9] * removed + appended
    prefix_len = len(prepared.parent_tokens) - removed
    edit = PromptEdit(
        removed_prefix_token_count=removed,
        input_delta=delta,
        routed_experts_start_len=min(prepared.reusable_r3_rows, prefix_len),
    )
    return edit, prepared.parent_tokens[:prefix_len] + delta


def _assistant(text):
    return {"role": "assistant", "content": text}


def _generation(text, edit, *, finish_reason="stop", logprob=-0.5, r3=None):
    tokens = [ord(c) for c in text] + [END]
    meta_info = {
        "output_token_logprobs": [[logprob, token] for token in tokens],
        "completion_tokens": len(tokens),
        "weight_version": 3,
    }
    return Generation(
        response={"id": f"resp-{text}"},
        message=_assistant(text),
        output_tokens=tokens,
        output_logprobs=[logprob] * len(tokens),
        finish_reason=finish_reason,
        meta_info=meta_info,
        prompt_edit=edit,
        r3=r3,
    )


class Session:
    """A manager plus the flat reference copy of every turn's full sequence."""

    def __init__(self, replace_tail=False):
        self.manager = TrajectoryManager()
        self.replace_tail = replace_tail
        self.flat: dict[int, list[int]] = {}

    def prepare(self, messages, key=KEY):
        return self.manager.prepare(messages, context_key=key)

    def prompt(self, prepared):
        return _render_like_sglang(prepared, replace_tail=self.replace_tail)[1]

    def commit(self, prepared, reply, **kwargs):
        edit, prompt = _render_like_sglang(prepared, replace_tail=self.replace_tail)
        generation = _generation(reply, edit, **kwargs)
        node = self.manager.commit(prepared, generation)
        self.flat[node.node_id] = prompt + generation.output_tokens
        return node

    def turn(self, messages, reply, **kwargs):
        return self.commit(self.prepare(messages), reply, **kwargs)

    def assert_matches_flat(self):
        for node_id, tokens in self.flat.items():
            assert self.manager.tokens(node_id) == tokens


def test_linear_turns_store_only_the_delta():
    s = Session()
    first = s.turn([SYSTEM, USER], "a1")
    tool = {"role": "tool", "content": "obs"}
    prepared = s.prepare([SYSTEM, USER, _assistant("a1"), tool])
    assert prepared.parent_id == first.node_id
    assert prepared.new_messages == [tool]
    second = s.commit(prepared, "a2")

    assert second.prefix_len == first.end
    assert second.input_delta == _render([tool]) + [GEN]
    assert second.delta_messages == [tool, _assistant("a2")]
    s.assert_matches_flat()


def test_rollback_retry_and_new_root_branch_without_touching_old_turns():
    s = Session()
    first = s.turn([SYSTEM, USER], "a1")
    history = [SYSTEM, USER, _assistant("a1")]
    second = s.turn([*history, {"role": "user", "content": "x"}], "a2")

    # Rollback: drop a2 and continue a1 differently -> sibling of turn 2.
    rollback = s.turn([*history, {"role": "user", "content": "y"}], "b2")
    # Retry: the same request as turn 2 again -> another sibling with the same prompt.
    retry = s.turn([*history, {"role": "user", "content": "x"}], "c2")
    # Compaction: nothing matches -> new root.
    root = s.turn([SYSTEM, {"role": "user", "content": "summary"}], "d1")

    assert rollback.parent_id == retry.parent_id == second.parent_id == first.node_id
    assert root.parent_id is None
    assert s.manager.tokens(second.node_id) == s.flat[second.node_id]
    s.assert_matches_flat()


def test_tools_or_template_kwargs_change_starts_a_new_root():
    s = Session()
    s.turn([SYSTEM, USER], "a1")
    messages = [SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "more"}]
    other_key = context_key([{"type": "function", "function": {"name": "f"}}], {})
    assert s.prepare(messages, key=other_key).parent_id is None
    assert s.prepare(messages, key=context_key(None, {"enable_thinking": False})).parent_id is None


def test_truncated_turn_is_never_a_parent():
    s = Session()
    first = s.turn([SYSTEM, USER], "a1")
    history = [SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}]
    s.turn(history, "cut", finish_reason="length")

    prepared = s.prepare([*history, _assistant("cut"), {"role": "user", "content": "continue"}])
    # Attaches before the cut turn; the cut text becomes context of the new turn.
    assert prepared.parent_id == first.node_id
    assert prepared.new_messages[1] == _assistant("cut")


def test_identical_replies_stay_separate_and_the_latest_is_extended():
    s = Session()
    s.turn([SYSTEM, USER], "a1")
    request = [SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}]
    # Prepared concurrently from the same parent, committed in either order: siblings.
    p1, p2 = s.prepare(request), s.prepare(request)
    twin_a = s.commit(p2, "same", logprob=-1.0)
    twin_b = s.commit(p1, "same", logprob=-2.0)
    assert twin_a.parent_id == twin_b.parent_id
    assert twin_a.output_logprobs != twin_b.output_logprobs

    prepared = s.prepare([*request, _assistant("same"), {"role": "user", "content": "next"}])
    assert prepared.parent_id == twin_b.node_id


def test_tail_replacement_keeps_parent_immutable():
    s = Session(replace_tail=True)
    first = s.turn([SYSTEM, USER], "a1")
    second = s.turn([SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}], "a2")
    assert second.prefix_len == first.end - 1
    assert s.manager.tokens(first.node_id)[-1] == END
    s.assert_matches_flat()

    (sample,) = s.manager.leaf_samples(max_seq_len=None, tokenizer=_Decoder())
    # a1's replaced END is gone from this path, so its output keeps loss on every token but the last.
    response = sample.tokens[-sample.response_length :]
    assert sample.loss_mask == [1 if i < 2 or i >= len(response) - 3 else 0 for i in range(len(response))]


def test_export_matches_flat_tokens_with_masks_and_logprobs():
    s = Session()
    first = s.turn([SYSTEM, USER], "a1", logprob=-0.1)
    tool = {"role": "tool", "content": "obs"}
    second = s.turn([SYSTEM, USER, _assistant("a1"), tool], "a2", logprob=-0.2)

    (sample,) = s.manager.leaf_samples(max_seq_len=None, tokenizer=_Decoder())
    sample.validate()
    assert sample.tokens == s.flat[second.node_id]
    assert sample.response_length == len(sample.tokens) - first.prompt_len
    out1, delta2, out2 = len(first.output_tokens), len(second.input_delta), len(second.output_tokens)
    assert sample.loss_mask == [1] * out1 + [0] * delta2 + [1] * out2
    assert sample.rollout_log_probs == [-0.1] * out1 + [0.0] * delta2 + [-0.2] * out2
    assert [call.spans[0].abs_start for call in sample.weight_versions] == [first.prompt_len, second.prompt_len]
    assert sample.metadata["leaf"]["path_node_ids"] == [0, 1]


def test_max_seq_len_cuts_at_turn_level():
    s = Session()
    first = s.turn([SYSTEM, USER], "a1")
    second = s.turn([SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}], "a2")

    (cut_in_output,) = s.manager.leaf_samples(max_seq_len=second.end - 1, tokenizer=_Decoder())
    assert cut_in_output.status == Sample.Status.TRUNCATED
    assert len(cut_in_output.tokens) == second.end - 1

    (cut_in_prompt,) = s.manager.leaf_samples(max_seq_len=second.prompt_len, tokenizer=_Decoder())
    assert cut_in_prompt.status == Sample.Status.COMPLETED
    assert cut_in_prompt.tokens == s.flat[first.node_id]

    assert s.manager.leaf_samples(max_seq_len=first.prompt_len, tokenizer=_Decoder()) == []


def test_routing_replay_rows_are_rebuilt_from_additional_chunks():
    s = Session()
    first_prepared = s.prepare([SYSTEM, USER])
    total_first = len(s.prompt(first_prepared)) + 3  # "a1" + END
    rows = np.arange(1000, dtype=np.int32).reshape(-1, 2, 1)  # synthetic [row, layer, topk]
    first = s.commit(first_prepared, "a1", r3=R3Chunk(row_start=0, values=rows[: total_first - 1]))

    prepared = s.prepare([SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}])
    assert prepared.reusable_r3_rows == first.end - 1
    total = len(s.prompt(prepared)) + 3
    start = prepared.reusable_r3_rows
    s.commit(prepared, "a2", r3=R3Chunk(row_start=start, values=rows[start : total - 1]))

    (sample,) = s.manager.leaf_samples(max_seq_len=None, tokenizer=_Decoder())
    np.testing.assert_array_equal(sample.rollout_routed_experts, rows[: total - 1])
    sample.validate()


def test_routing_replay_gap_is_an_error():
    s = Session()
    first_prepared = s.prepare([SYSTEM, USER])
    rows = np.ones((100, 2, 1), dtype=np.int32)
    s.commit(first_prepared, "a1", r3=R3Chunk(row_start=0, values=rows[:3]))  # too few rows
    prepared = s.prepare([SYSTEM, USER, _assistant("a1"), {"role": "user", "content": "x"}])
    s.commit(prepared, "a2", r3=R3Chunk(row_start=prepared.reusable_r3_rows, values=rows[:50]))
    with pytest.raises(ValueError, match="routing rows"):
        s.manager.leaf_samples(max_seq_len=None, tokenizer=_Decoder())


def test_miles_hooks_drop_retried_leaf_and_mask_shared_turn_once():
    s = Session()
    s.turn([SYSTEM, USER], "a1")
    history = [SYSTEM, USER, _assistant("a1")]
    s.turn([*history, {"role": "user", "content": "x"}], "a2")  # superseded by the retry below
    s.turn([*history, {"role": "user", "content": "x"}], "b2")
    s.turn([*history, {"role": "user", "content": "y"}], "c2")  # a real second branch

    metadata = {"tree": s.manager.tree_metadata(), "agent": {"reward": 1.0}}
    samples = s.manager.leaf_samples(max_seq_len=None, tokenizer=_Decoder())
    kept = default_postprocess(drop_same_prompt_retries(samples, metadata), metadata)

    assert sorted(sample.metadata["leaf"]["node_id"] for sample in kept) == [2, 3]
    shared_turn_loss = [sum(sample.loss_mask[:3]) for sample in kept]  # a1's 3 output tokens
    assert sorted(shared_turn_loss) == [0, 3]
    assert all(sample.reward == 1.0 for sample in kept)


def test_fingerprint_ignores_wire_only_differences():
    call = {"id": "a", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"x": 1, "y": 2}'}}
    replayed = {"id": "b", "type": "function", "function": {"name": "f", "arguments": '{"y":2,"x":1}'}}
    stored = {"role": "assistant", "content": None, "reasoning_content": "r", "tool_calls": [call]}
    assert message_fingerprint(stored) == message_fingerprint(
        {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [replayed], "extra": 1}
    )
    assert message_fingerprint({"role": "user", "content": [{"type": "text", "text": "hi"}]}) == message_fingerprint(
        {"role": "user", "content": "hi"}
    )
    assert message_fingerprint({**stored, "content": "\n\n"}) == message_fingerprint(stored)
    edit = {"id": "e", "type": "function", "function": {"name": "Edit", "arguments": '{"old": "a", "new": "b"}'}}
    filled = {**edit, "function": {"name": "Edit", "arguments": '{"old": "a", "new": "b", "replace_all": false}'}}
    renamed = {**edit, "function": {"name": "Write", "arguments": '{"old": "a", "new": "b"}'}}
    as_message = lambda call: {"role": "assistant", "content": "", "tool_calls": [call]}  # noqa: E731
    assert message_fingerprint(as_message(edit)) == message_fingerprint(as_message(filled))
    assert message_fingerprint(as_message(edit)) != message_fingerprint(as_message(renamed))
    system = {"role": "system", "content": "Be useful.\n<total_tokens>15000000 tokens left</total_tokens>"}
    later = {"role": "system", "content": system["content"] + "\n<total_tokens>14984558 tokens left</total_tokens>"}
    assert message_fingerprint(system) == message_fingerprint(later)
    assert message_fingerprint(system) != message_fingerprint({"role": "system", "content": "Be careful."})
    assert message_fingerprint(stored) != message_fingerprint({**stored, "reasoning_content": None})


class _Decoder:
    def decode(self, tokens):
        return " ".join(map(str, tokens))
