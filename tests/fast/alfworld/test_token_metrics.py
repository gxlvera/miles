from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.experimental.alfworld.token_metrics import _summarize, turn_token_counts
from examples.experimental.alfworld import token_metrics


def sample(mask, prompt_length, turns, index=0):
    return SimpleNamespace(tokens=list(range(prompt_length + len(mask))), response_length=len(mask),
                           loss_mask=mask, metadata={"num_turns": turns, "task_success": True}, index=index)


def test_merged_turn_counts_include_initial_prompt_and_tool_context_without_mutation():
    s = sample([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3, 3)
    before = deepcopy(s.__dict__)
    rows = turn_token_counts(s)
    assert [r['assistant_tokens'] for r in rows] == [2, 3, 4]
    assert [r['non_assistant_new_tokens'] for r in rows] == [3, 3, 3]
    assert [r['non_assistant_context_tokens'] for r in rows] == [3, 6, 9]
    assert [r['prompt_context_tokens'] for r in rows] == [3, 8, 14]
    assert sum(r['assistant_tokens'] + r['non_assistant_new_tokens'] for r in rows) == len(s.tokens)
    assert s.__dict__ == before


def test_batch_mean_weights_turns_equally_and_reports_turn_positions():
    a = sample([1, 1], 3, 1)
    b = sample([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 5, 2, index=1)
    metrics, _ = _summarize([a, b])
    assert metrics['per_turn/assistant_tokens/mean'] == 4
    assert metrics['per_turn/non_assistant_new_tokens/mean'] == 3
    assert metrics['turn_02/count'] == 1
    assert metrics['episode/turns/mean'] == 1.5


def test_mismatched_turns_fail_instead_of_silently_logging_wrong_lengths():
    with pytest.raises(ValueError, match='completion span'):
        turn_token_counts(sample([1, 1], 3, 2))


def test_eval_logs_weighted_turn_means_without_position_series(monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(token_metrics.tracking, "log", lambda args, data, **kw: logged.append(data))
    args = SimpleNamespace(dump_details=str(tmp_path / "dump_details"), wandb_always_use_train_step=False)
    samples = [sample([1, 1], 3, 1), sample([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 5, 2, index=1)]
    assert token_metrics.log_eval(25, args, {"holdout": {"samples": samples}}, {}) is False
    metrics = logged[0]
    assert metrics["eval/step"] == 25
    assert metrics["eval/holdout/tokens/per_turn/assistant_tokens/mean"] == 4
    assert metrics["eval/holdout/tokens/per_turn/non_assistant_new_tokens/mean"] == 3
    assert metrics["eval/holdout/tokens/episode/success_rate"] == 1
    assert not any("/turn_" in key for key in metrics)
    assert len(list((tmp_path / "token_metrics").glob("*.json"))) == 1
