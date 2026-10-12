"""Read-only measurements; the supplied per-turn implementation is unchanged."""
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import mean

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    episodes = defaultdict(list)
    for sample in samples:
        episodes[sample.rollout_id if sample.rollout_id is not None else sample.index].append(sample)
    rows = []
    for episode_id, parts in episodes.items():
        split = all(s.metadata.get("sample_mode") == "per-turn" for s in parts)
        last = max(parts, key=lambda s: len(s.tokens))
        if split:
            merged_tokens = len(last.tokens)
            per_turn_tokens = sum(len(s.tokens) for s in parts)
            turns = len(parts)
        else:
            assert len(parts) == 1, "Expected one append-only trajectory per episode"
            position = len(last.tokens) - last.response_length
            ends = []
            for enabled, values in groupby(last.loss_mask):
                position += sum(1 for _ in values)
                if enabled:
                    ends.append(position)
            merged_tokens, per_turn_tokens, turns = len(last.tokens), sum(ends), len(ends)
        rows.append(dict(episode_id=episode_id, group_index=last.group_index,
                         task_id=last.metadata["instance_id"], samples=len(parts), turns=turns,
                         merged_tokens=merged_tokens, per_turn_tokens=per_turn_tokens,
                         actual_tokens=sum(len(s.tokens) for s in parts),
                         loss_tokens=sum(sum(s.loss_mask) for s in parts), reward=last.reward,
                         branch_selection=last.metadata.get("branch_selection", {})))
    total = sum(r["actual_tokens"] for r in rows)
    merged = sum(r["merged_tokens"] for r in rows)
    per_turn = sum(r["per_turn_tokens"] for r in rows)
    prompts = len({r["group_index"] for r in rows})
    metrics = dict(requests=len(rows), prompt_groups=prompts, episodes=len(rows), samples=len(samples),
                   samples_per_request=len(samples)/len(rows), samples_per_prompt_group=len(samples)/prompts,
                   samples_per_episode=len(samples)/len(rows),
                   tokens=total, merged_tokens=merged, per_turn_tokens=per_turn,
                   duplicate_tokens=total-merged, duplicate_fraction=(total-merged)/total,
                   counterfactual_per_turn_duplicate_fraction=(per_turn-merged)/per_turn,
                   token_expansion=per_turn/merged, loss_tokens=sum(r["loss_tokens"] for r in rows),
                   reward=mean(r["reward"] for r in rows), rollout_seconds=rollout_time)
    metrics.update({key: sum(r["branch_selection"].get(key, 0) for r in rows) for key in
                    ("leaves", "total_turns", "selected_turns", "discarded_leaves", "discarded_unique_turns",
                     "discarded_unique_completion_tokens", "discarded_leaf_sequence_tokens")})
    output = Path(args.dump_details).parent / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    (output/f"rollout_{rollout_id}.json").write_text(json.dumps(dict(metrics=metrics, episodes=rows), indent=2))
    tracking.log(args, {**{f"ablation/{k}":v for k,v in metrics.items()},
                        "rollout/step":compute_rollout_step(args, rollout_id)}, step_key="rollout/step")
    print("ABLATION_METRICS="+json.dumps(metrics), flush=True)
    return False
