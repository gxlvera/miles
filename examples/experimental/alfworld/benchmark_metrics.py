"""Compare merged and per-turn ALFWorld training layouts without changing samples.

Token counts include complete unpadded token sequences before the trainer's
causal shift. Duplicate history means repeated positions within one episode,
not repeated vocabulary or shared prefixes between independent episodes.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def summarize_batch(samples):
    if not samples:
        raise ValueError("Cannot benchmark an empty batch")
    episodes = defaultdict(list)
    for sample in samples:
        episode_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        if sample.group_index is None or episode_id is None:
            raise ValueError("Benchmark requires prompt group and episode identities")
        if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
            raise ValueError("Benchmark requires a response-aligned loss mask")
        if any(value not in (0, 1) for value in sample.loss_mask):
            raise ValueError("Benchmark requires a binary loss mask")
        episodes[(sample.group_index, episode_id)].append(sample)

    unique_tokens = 0
    for siblings in episodes.values():
        final = max(siblings, key=lambda sample: len(sample.tokens))
        per_turn = final.metadata.get("sample_mode") == "per-turn"
        if per_turn:
            turn_indices = sorted(s.metadata["turn_index"] for s in siblings)
            if turn_indices != list(range(1, final.metadata["num_turns"] + 1)):
                raise ValueError("Per-turn benchmark requires every turn exactly once")
            if any(s.metadata.get("sample_mode") != "per-turn" for s in siblings):
                raise ValueError("Mixed sample layouts within one episode")
            if any(s.tokens != final.tokens[:len(s.tokens)] for s in siblings):
                raise ValueError("Duplicate-history accounting requires append-only prefixes")
        elif len(siblings) != 1:
            raise ValueError("Merged benchmark requires one sample per episode")
        unique_tokens += len(final.tokens)

    token_count = sum(len(sample.tokens) for sample in samples)
    if token_count == 0:
        raise ValueError("Cannot benchmark empty token sequences")
    prompt_count = len({key[0] for key in episodes})
    duplicate_tokens = token_count - unique_tokens
    return {
        "source_prompts": prompt_count,
        "episodes": len(episodes),
        "samples": len(samples),
        "samples_per_source_prompt": len(samples) / prompt_count,
        "samples_per_episode": len(samples) / len(episodes),
        "tokens_total": token_count,
        "tokens_unique_within_episodes": unique_tokens,
        "tokens_repeated_history": duplicate_tokens,
        "repeated_history_fraction": duplicate_tokens / token_count,
        "assistant_trainable_tokens": sum(sum(s.loss_mask) for s in samples),
    }


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    metrics = summarize_batch(samples)
    if metrics["source_prompts"] != args.rollout_batch_size:
        raise ValueError("Benchmark prompt count differs from configured batch")
    if metrics["episodes"] != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("Benchmark episode count differs from configured batch")
    metrics["rollout_seconds"] = rollout_time
    step = compute_rollout_step(args, rollout_id)
    tracking.log(args, {"rollout/step": step, **{f"benchmark/{k}": v for k, v in metrics.items()}},
                 step_key="rollout/step")
    output = Path(args.dump_details).parent / "benchmark_metrics"
    output.mkdir(parents=True, exist_ok=True)
    record = {"rollout_id": rollout_id, "step": step, "metrics": metrics}
    (output / f"{rollout_id:06d}.json").write_text(json.dumps(record, indent=2))
    logger.info("ALFWorld benchmark %s: %s", rollout_id, metrics)
    return False  # Preserve native perf/train_time and perf/step_time metrics.
