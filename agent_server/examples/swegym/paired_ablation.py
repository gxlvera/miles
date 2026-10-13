"""The per-turn vs. turn-merge ablation: one rollout batch, frozen once, trained in either layout."""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from original_per_turn import split_per_turn

from miles.rollout.base_types import BaseRolloutFn, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.types import Sample

ROOT = Path("/workspace/proxy-turn-ablation/paired-data")


def loss_pairs(sample):
    tokens = sample.tokens[len(sample.tokens) - sample.response_length :]
    return [
        (token, logprob)
        for token, logprob, mask in zip(tokens, sample.rollout_log_probs, sample.loss_mask, strict=True)
        if mask
    ]


def pack(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        assert sample.group_index is not None and sample.rollout_id is not None
        groups[sample.group_index][sample.rollout_id].append(sample)
    return [[parts for parts in episodes.values()] for episodes in groups.values()]


def freeze_pair(merged, tokenizer, rollout_seconds):
    """Save the merged trajectories and their per-turn split, after checking that both train on the same tokens."""
    started = time.perf_counter()
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / "manifest.json").exists(), "Never overwrite the frozen pair"
    parts, rows = [], []
    for sample in merged:
        assert sample.status != Sample.Status.ABORTED
        sample.rollout_id = sample.index if sample.rollout_id is None else sample.rollout_id
        sample.validate()
        turns = split_per_turn(sample, tokenizer)
        assert turns[-1].tokens == sample.tokens, "a trajectory must end with a completion"
        assert all(turn.tokens == sample.tokens[: len(turn.tokens)] for turn in turns)
        assert all(turn.reward == sample.reward and turn.group_index == sample.group_index for turn in turns)
        assert [pair for turn in turns for pair in loss_pairs(turn)] == loss_pairs(sample)
        assert [v for turn in turns for v in turn.weight_versions] == sample.weight_versions
        assert sample.weight_versions and all(str(span.version).isdigit() for span in sample.all_weight_version_spans)
        assert sum(span.abs_end - span.abs_start for span in sample.all_weight_version_spans) == sum(sample.loss_mask)
        parts.extend(turns)
        rows.append(
            dict(
                episode_id=sample.rollout_id,
                group_index=sample.group_index,
                task_id=sample.metadata["instance_id"],
                turns=len(turns),
                merged_tokens=len(sample.tokens),
                per_turn_tokens=sum(len(t.tokens) for t in turns),
                loss_tokens=sum(sample.loss_mask),
                reward=sample.reward,
                branch_selection=sample.metadata["branch_selection"],
            )
        )
    assert len({s.rollout_id for s in merged}) == len(merged)
    manifest = dict(
        episodes=rows,
        requests=len(merged),
        prompt_groups=len({s.group_index for s in merged}),
        per_turn_samples=len(parts),
        merged_samples=len(merged),
        per_turn_tokens=sum(len(s.tokens) for s in parts),
        merged_tokens=sum(len(s.tokens) for s in merged),
        loss_tokens=sum(sum(s.loss_mask) for s in merged),
        shared_rollout_seconds=rollout_seconds,
        exact_loss_token_logprob_conservation=True,
        exact_weight_version_conservation=True,
        full_prefix_conservation=True,
        splitter_sha256=hashlib.sha256(Path(__file__).with_name("original_per_turn.py").read_bytes()).hexdigest(),
    )
    for name, samples in [("merged", merged), ("per-turn", parts)]:
        path = ROOT / f"{name}.pt"
        torch.save({"samples": [s.to_dict() for s in samples]}, path)
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        manifest[f"{name}_sha256"] = digest
    manifest["preparation_seconds"] = time.perf_counter() - started
    manifest["duplicate_prefix_tokens"] = manifest["per_turn_tokens"] - manifest["merged_tokens"]
    manifest["duplicate_prefix_fraction"] = manifest["duplicate_prefix_tokens"] / manifest["per_turn_tokens"]
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FROZEN_PAIR=" + json.dumps({k: v for k, v in manifest.items() if k != "episodes"}), flush=True)
    return parts


class Baseline(BaseRolloutFn):
    async def __call__(self, input):
        args = self.constructor_input.args
        assert args.num_rollout == 1 and input.rollout_id == 0 and not input.evaluation
        assert not args.colocate and not args.fully_async and args.use_agent_server
        if (ROOT / "manifest.json").exists():
            return replay(args, "per-turn")
        state = GenerateState(args)
        groups = self.constructor_input.data_source.get_samples(args.rollout_batch_size)
        assert sum(map(len, groups)) == args.global_batch_size
        started = time.perf_counter()
        ROOT.mkdir(parents=True, exist_ok=True)
        output = await asyncio.gather(*[collect_group(state, group) for group in groups])
        rollout_seconds = time.perf_counter() - started
        merged = []
        for group in output:
            for episode in group:
                assert isinstance(episode, list) and len(episode) == 1, "the picker must select one leaf per episode"
                merged.extend(episode)
        torch.save(
            {"samples": [s.to_dict() for s in merged], "rollout_seconds": rollout_seconds}, ROOT / "collected.pt"
        )
        parts = freeze_pair(merged, state.tokenizer, rollout_seconds)
        return RolloutFnTrainOutput(samples=pack(parts), metrics={"paired_live_collection": 1})


async def collect_group(state, group):
    output = await generate_and_rm_group(state, group, state.sampling_params)
    torch.save(output, ROOT / f"collected-group-{group[0].group_index}.pt")
    print(f"GROUP_COLLECTED group={group[0].group_index} episodes={len(output)}", flush=True)
    return output


def replay(args, mode):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["requests"] == args.global_batch_size
    path = ROOT / f"{mode}.pt"
    assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == manifest[f"{mode}_sha256"]
    payload = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(row) for row in payload["samples"]]
    return RolloutFnTrainOutput(samples=pack(samples), metrics={"paired_live_collection": 0})


class MergedReplay(BaseRolloutFn):
    def __call__(self, input):
        args = self.constructor_input.args
        assert args.num_rollout == 1 and input.rollout_id == 0 and not input.evaluation
        assert not args.colocate and not args.debug_train_only
        return replay(args, "merged")


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from ablation_metrics import log_rollout as native_log

    from miles.utils.metric_utils import compute_rollout_step
    from miles.utils.tracking_utils import tracking

    manifest = json.loads((ROOT / "manifest.json").read_text())
    native_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    metrics = dict(
        shared_rollout_seconds=manifest["shared_rollout_seconds"],
        pair_preparation_seconds=manifest["preparation_seconds"],
        longest_trajectory_only=1,
        exact_pair_conservation=1,
    )
    tracking.log(
        args,
        {**{f"ablation/{k}": v for k, v in metrics.items()}, "rollout/step": compute_rollout_step(args, rollout_id)},
        step_key="rollout/step",
    )
    return False
