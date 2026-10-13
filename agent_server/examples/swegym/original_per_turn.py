"""The per-turn baseline of the per-turn vs. turn-merge ablation: one sample per model turn.

A merged trajectory marks each completion with ones and tool observations with zeros, so each
contiguous run of ones is one model turn. Each turn's sample keeps the whole history before it.
"""

from dataclasses import replace
from itertools import groupby

from miles.utils.types import Sample


def split_per_turn(sample: Sample, tokenizer) -> list[Sample]:
    """Preserve token IDs/logprobs and episode identity; change training layout only."""
    if sample.status == Sample.Status.ABORTED:
        return [sample]  # Leave native retry/filter behavior intact.
    # Qwen3.5-4B is dense and this recipe uses plain GRPO. Reject side channels
    # rather than accidentally copy an unsliced tensor into each turn.
    for field in (
        "multimodal_inputs",
        "multimodal_train_inputs",
        "rollout_routed_experts",
        "rollout_indexer_topk",
        "rollout_sampling_mask",
        "rollout_topk_token_ids",
        "rollout_topk_log_probs",
        "teacher_log_probs",
        "opd_reverse_kl",
    ):
        if getattr(sample, field) is not None:
            raise ValueError(f"the per-turn split does not support {field}")
    sample.validate()
    if sample.loss_mask is None or sample.rollout_log_probs is None:
        raise ValueError("Expected loss masks and rollout logprobs")
    if any(value not in (0, 1) for value in sample.loss_mask):
        raise ValueError("Expected a binary completion/observation mask")

    spans = []
    offset = 0
    for enabled, values in groupby(sample.loss_mask):
        end = offset + sum(1 for _ in values)
        if enabled:
            spans.append((offset, end))
        offset = end
    # Detect compaction, partial collection, or a changed masking convention.
    # The current harness has no branches, no zero-token turns and no max_seq_len.
    if not spans or len(spans) != sample.metadata.get("num_turns"):
        raise ValueError("Expected one completion span per model turn (a trajectory not cut by max_seq_len)")
    if sample.weight_versions and len(sample.weight_versions) != len(spans):
        raise ValueError("Expected one weight-version record per model turn")
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    if rollout_id is None:
        raise ValueError("Per-turn samples need an episode index or rollout_id")

    response_start = len(sample.tokens) - sample.response_length
    turns = []
    for i, (left, right) in enumerate(spans):
        start, end = response_start + left, response_start + right
        tokens = sample.tokens[:end]
        # Keep the full chat dump only on the last turn, where it is accurate.
        metadata = dict(sample.metadata)
        if i < len(spans) - 1:
            metadata.pop("messages", None)
        if isinstance(metadata.get("lifecycle"), list):
            metadata["lifecycle"] = metadata["lifecycle"][i : i + 1]
        metadata.update(sample_mode="per-turn", turn_index=i + 1, accumulated_token_ids=tokens)
        turn = replace(
            sample,
            rollout_id=rollout_id,
            tokens=tokens,
            # Decode for diagnostics only. Training uses the original token IDs,
            # never decode/re-encode or re-template the stored conversation.
            prompt=tokenizer.decode(sample.tokens[:start]),
            response=tokenizer.decode(sample.tokens[start:end]),
            response_length=right - left,
            loss_mask=sample.loss_mask[left:right],
            rollout_log_probs=sample.rollout_log_probs[left:right],
            weight_versions=sample.weight_versions[i : i + 1],
            status=sample.status if i == len(spans) - 1 else Sample.Status.COMPLETED,
            metadata=metadata,
            # These are episode totals; count each once, not once per sample.
            non_generation_time=sample.non_generation_time if i == 0 else 0.0,
            spec_info=sample.spec_info if i == 0 else Sample.SpecInfo(),
            prefix_cache_info=sample.prefix_cache_info if i == 0 else Sample.PrefixCacheInfo(),
        )
        turn.validate()
        turns.append(turn)
    return turns
