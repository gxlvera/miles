from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.experimental.alfworld.per_turn import split_per_turn
from scripts.run_qwen3_5_4b_alfworld import ScriptArgs, execute

from miles.backends.training_utils.data import context_parallel
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import (
    can_schedule_on_rollout_side,
    convert_samples_to_train_data,
    split_train_data_by_dp_scheduled_raw,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class Tokenizer:
    def decode(self, tokens):
        return " ".join(map(str, tokens))


def episode(*, index=10, group_index=7, reward=1, num_turns=3, truncated=False):
    """Native per-call Samples -> native merge -> the baseline's input."""
    turns = []
    tokens = [10, 11, 12]
    for i in range(num_turns):
        if i:
            tokens = tokens + [20, 21, 22]  # tool result and chat delimiters
        start = len(tokens)
        output = list(range(30 + 10 * i, 32 + 11 * i))
        tokens = tokens + output
        turns.append(
            Sample(
                index=index,
                group_index=group_index,
                tokens=tokens,
                prompt=Tokenizer().decode(tokens[:start]),
                response=Tokenizer().decode(output),
                response_length=len(output),
                loss_mask=[1] * len(output),
                rollout_log_probs=[-0.01 * value for value in output],
                weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(i + 1), start, len(tokens))])],
                reward=reward,
                metadata={"num_turns": num_turns},
                status=Sample.Status.TRUNCATED if truncated and i == num_turns - 1 else Sample.Status.COMPLETED,
                non_generation_time=1.25,
                prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=start - 1, total_prompt_tokens=start),
            )
        )
    return turns, merge_samples(turns, Tokenizer())


@pytest.mark.parametrize("truncated", [False, True])
def test_native_merge_round_trip_preserves_every_training_token(truncated):
    originals, merged = episode(truncated=truncated)
    merged.metadata["messages"] = [{"role": "assistant", "content": "last turn"}]
    merged.metadata["lifecycle"] = [{"turn": i + 1} for i in range(len(originals))]
    before = deepcopy(merged.to_dict())
    actual = split_per_turn(merged, Tokenizer())
    for original, turn in zip(originals, actual, strict=True):
        for field in (
            "tokens",
            "prompt",
            "response",
            "response_length",
            "loss_mask",
            "rollout_log_probs",
            "weight_versions",
            "status",
            "reward",
            "group_index",
            "index",
        ):
            assert getattr(turn, field) == getattr(original, field), field
        assert turn.rollout_id == merged.index
        assert turn.metadata["accumulated_token_ids"] == turn.tokens
    assert sum(turn.effective_response_length for turn in actual) == merged.effective_response_length
    assert sum(len(turn.tokens) for turn in actual) > len(merged.tokens)
    assert sum(turn.non_generation_time for turn in actual) == merged.non_generation_time
    assert sum(turn.prefix_cache_info.cached_tokens for turn in actual) == merged.prefix_cache_info.cached_tokens
    assert merged.to_dict() == before
    assert "messages" not in actual[0].metadata
    assert actual[-1].metadata["messages"] == merged.metadata["messages"]
    assert [s.metadata["lifecycle"] for s in actual] == [[{"turn": i + 1}] for i in range(3)]


@pytest.mark.parametrize("dp_size", [1, 2])
def test_native_training_path_keeps_one_step_and_the_same_episode_objective(monkeypatch, dp_size):
    # Unequal episode lengths expose accidental turn-weighted reward normalization.
    episodes = [episode(index=i, group_index=i // 2, reward=i % 2, num_turns=n)[1] for i, n in enumerate([1, 3, 2, 2])]
    args = SimpleNamespace(
        global_batch_size=4,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
        rollout_top_logprobs_num=0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=20,
        balance_data=True,
    )
    parallel = dict(dp_size=dp_size, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    results = []
    for per_turn in (False, True):
        groups = [
            [split_per_turn(s, Tokenizer()) if per_turn else [s] for s in episodes[start : start + 2]]
            for start in (0, 2)
        ]
        samples, metadata = postprocess_rollout_data(args, groups, parallel)
        assert len(samples) == (8 if per_turn else 4)  # no trailing turns trimmed
        data = convert_samples_to_train_data(args, samples, metadata, None, None)
        assert can_schedule_on_rollout_side(args, data, parallel)
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=parallel)
        assert all(shard["num_rollouts"] == [4] for shard in shards)  # exactly one optimizer step
        assert sum(len(shard["tokens"]) for shard in shards) == len(samples)
        # Use Miles' actual per-rollout loss reducer, with a differentiable toy
        # token loss, to check both the objective and gradient's normalization.
        reducer = context_parallel.get_sum_of_sample_mean(
            [len(s.tokens) for s in samples],
            data["response_lengths"],
            [torch.tensor(mask) for mask in data["loss_masks"]],
            denominators=torch.tensor(data["rollout_mask_sums"]),
        )
        scale = torch.tensor(0.7, requires_grad=True)
        losses = torch.cat(
            [
                torch.tensor(s.rollout_log_probs) * reward * scale
                for s, reward in zip(samples, data["rewards"], strict=True)
            ]
        )
        loss = reducer(losses) / 4
        loss.backward()
        results.append((data, loss.item(), scale.grad.item()))
    merged, split = results
    assert split[1:] == pytest.approx(merged[1:])
    for rid in range(4):
        positions = [i for i, value in enumerate(split[0]["rollout_ids"]) if value == rid]
        assert all(split[0]["rewards"][i] == merged[0]["rewards"][rid] for i in positions)
        assert all(split[0]["rollout_mask_sums"][i] == merged[0]["rollout_mask_sums"][rid] for i in positions)


def test_aborted_episode_is_left_to_native_filtering():
    aborted = Sample(index=5, status=Sample.Status.ABORTED)
    assert split_per_turn(aborted, Tokenizer()) == [aborted]


def test_changed_turn_layout_and_replay_data_fail_loudly():
    _, sample = episode()
    sample.metadata["num_turns"] += 1
    with pytest.raises(ValueError, match="completion span"):
        split_per_turn(sample, Tokenizer())
    sample.rollout_routed_experts = torch.zeros(1)
    with pytest.raises(ValueError, match="rollout_routed_experts"):
        split_per_turn(sample, Tokenizer())


def test_launcher_switch_changes_only_the_generation_adapter(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(ScriptArgs, "create_backend", lambda self: backend)
    assert ScriptArgs().per_turn_samples is False
    for enabled in (False, True):
        execute(ScriptArgs(per_turn_samples=enabled, run_id="same-run-for-command-comparison"))
    merged, per_turn = [call.kwargs for call in backend.execute_train.call_args_list]
    assert "--global-batch-size 8" in per_turn["train_args"]
    assert per_turn["train_args"] == merged["train_args"].replace(
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        "examples.experimental.alfworld.per_turn.generate",
    )
    assert per_turn["extra_env_vars"] == merged["extra_env_vars"]
