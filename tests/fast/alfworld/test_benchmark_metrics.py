from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.experimental.alfworld.benchmark_metrics import summarize_batch


def sample(tokens, mask, index=0, group=0, turn=None):
    return SimpleNamespace(tokens=tokens, loss_mask=mask, response_length=len(mask), index=index,
                           group_index=group, rollout_id=index if turn else None,
                           metadata={"sample_mode": "per-turn", "turn_index": turn, "num_turns": 3} if turn else {})


def test_per_turn_repeated_prefix_positions_and_merged_trainable_tokens_agree():
    tokens = list(range(18))
    turns = [sample(tokens[:5], [1, 1], turn=1), sample(tokens[:11], [1] * 3, turn=2),
             sample(tokens, [1] * 4, turn=3)]
    before = deepcopy(turns)
    per_turn = summarize_batch(turns)
    merged = summarize_batch([sample(tokens, [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])])
    assert per_turn["tokens_total"] == 34
    assert per_turn["tokens_unique_within_episodes"] == 18
    assert per_turn["tokens_repeated_history"] == 16
    assert per_turn["repeated_history_fraction"] == pytest.approx(16 / 34)
    assert per_turn["samples_per_source_prompt"] == 3
    assert per_turn["samples_per_episode"] == 3
    assert merged["repeated_history_fraction"] == 0
    assert merged["tokens_total"] == 18
    assert per_turn["assistant_trainable_tokens"] == merged["assistant_trainable_tokens"] == 9
    assert [s.__dict__ for s in turns] == [s.__dict__ for s in before]


def test_prompts_episodes_and_lexical_repetition_are_distinct():
    samples = [sample([1] * 5, [1, 1], index=i, group=g) for g in range(4) for i in range(8)]
    metrics = summarize_batch(samples)
    assert metrics["source_prompts"] == 4
    assert metrics["episodes"] == metrics["samples"] == 32
    assert metrics["samples_per_source_prompt"] == 8
    assert metrics["samples_per_episode"] == 1
    assert metrics["tokens_total"] == 160
    assert metrics["tokens_repeated_history"] == 0


def test_reject_missing_turns_and_non_prefix_histories():
    with pytest.raises(ValueError, match="every turn"):
        summarize_batch([sample([1, 2], [1], turn=1)])
    with pytest.raises(ValueError, match="append-only"):
        summarize_batch([sample([9], [1], turn=1), sample([1, 2], [1], turn=2),
                         sample([1, 2, 3], [1], turn=3)])
