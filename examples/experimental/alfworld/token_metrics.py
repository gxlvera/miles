"""Read-only per-turn diagnostics for native merged ALFWorld samples.

Assistant counts use loss-mask ones. Non-assistant counts include initial
system/user/tool-schema context and later tool observations/chat delimiters.
Both incremental (new this turn) and accumulated context counts are recorded.
The final environment reply after the last model call is not a model input and
is not present in the merged training sample.
"""

import json
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import mean

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def turn_token_counts(sample):
    """Recover completion boundaries without splitting or re-tokenizing a sample."""
    mask = sample.loss_mask
    if mask is None or len(mask) != sample.response_length:
        raise ValueError("ALFWorld token metrics require the native response loss mask")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("ALFWorld token metrics require a binary loss mask")
    prompt_length = len(sample.tokens) - sample.response_length
    if prompt_length < 0:
        raise ValueError("Response length exceeds sample length")
    cursor, previous_end, non_assistant_context = prompt_length, 0, 0
    turns = []
    for enabled, values in groupby(mask):
        end = cursor + sum(1 for _ in values)
        if enabled:
            new_context = cursor - previous_end
            non_assistant_context += new_context
            turns.append({
                "turn": len(turns) + 1,
                "assistant_tokens": end - cursor,
                "non_assistant_new_tokens": new_context,
                "non_assistant_context_tokens": non_assistant_context,
                "prompt_context_tokens": cursor,
            })
            previous_end = end
        cursor = end
    if len(turns) != sample.metadata.get("num_turns"):
        raise ValueError("Expected one completion span per agent turn in a merged episode")
    if not turns or previous_end != len(sample.tokens):
        raise ValueError("Expected a merged episode ending at its final completion")
    return turns


def _summarize(samples, *, include_turn_positions=True):
    rows = [{"episode_index": s.index, "turns": turn_token_counts(s)} for s in samples]
    turns = [turn for row in rows for turn in row["turns"]]
    by_turn = defaultdict(list)
    for turn in turns:
        by_turn[turn["turn"]].append(turn)
    metrics = {"episodes": len(rows), "turns": len(turns)}
    fields = ("assistant_tokens", "non_assistant_new_tokens", "non_assistant_context_tokens", "prompt_context_tokens")
    for field in fields:
        metrics[f"per_turn/{field}/mean"] = mean(turn[field] for turn in turns)
    if include_turn_positions:
        for number, values in sorted(by_turn.items()):
            metrics[f"turn_{number:02d}/count"] = len(values)
            for field in fields:
                metrics[f"turn_{number:02d}/{field}/mean"] = mean(turn[field] for turn in values)
    metrics["episode/turns/mean"] = mean(len(row["turns"]) for row in rows)
    metrics["episode/merged_tokens/mean"] = mean(len(sample.tokens) for sample in samples)
    metrics["episode/success_rate"] = mean(float(sample.metadata["task_success"]) for sample in samples)
    return metrics, rows


def _record(rollout_id, args, samples, namespace, step_key, *, include_turn_positions=True):
    metrics, rows = _summarize(samples, include_turn_positions=include_turn_positions)
    log_dict = {f"{namespace}/{key}": value for key, value in metrics.items()}
    log_dict[step_key] = compute_rollout_step(args, rollout_id)
    tracking.log(args, log_dict, step_key=step_key)
    logger.info(
        "ALFWorld token lengths [%s] rollout=%s episodes=%s turns=%s assistant/turn=%.2f "
        "new_non_assistant/turn=%.2f accumulated_non_assistant/turn=%.2f success=%.4f",
        namespace, rollout_id, len(samples), metrics["turns"],
        metrics["per_turn/assistant_tokens/mean"], metrics["per_turn/non_assistant_new_tokens/mean"],
        metrics["per_turn/non_assistant_context_tokens/mean"], metrics["episode/success_rate"],
    )
    output = Path(args.dump_details).parent / "token_metrics"
    output.mkdir(parents=True, exist_ok=True)
    # Eval-before-train and eval-after-step can share a rollout ID: preserve both.
    stem = f"{namespace.replace('/', '_')}_{rollout_id:06d}"
    path = output / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = output / f"{stem}_{suffix}.json"
        suffix += 1
    path.write_text(json.dumps({"rollout_id": rollout_id, "namespace": namespace, "metrics": metrics, "episodes": rows}, indent=2))


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    _record(rollout_id, args, samples, "rollout/alfworld", "rollout/step")
    return False  # Retain all standard Miles metrics.


def log_eval(rollout_id, args, data, extra_metrics):
    for name, result in data.items():
        # Eval charts show means across all turns, not one series per turn index.
        # Raw per-episode counts remain in the local diagnostic JSON only.
        _record(rollout_id, args, result["samples"], f"eval/{name}/tokens", "eval/step",
                include_turn_positions=False)
    return False
