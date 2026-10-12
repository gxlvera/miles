# Opt-in Proxy per-turn / merged comparison

Use the environment, prepared data, authenticated gateway, and initial model
from [the SWE-Gym recipe](README.md). Both arms use the same initial weights.
The default launcher and all e2e runtime files are unchanged by this addition.

The switch is the existing `--rollout-function-path` extension point:

- `paired_proxy.Baseline` collects one batch through `agent_proxy`, selects the
  longest native trajectory per session with the recipe's existing picker,
  freezes it, and calls `original_per_turn.split_per_turn`.
- `paired_proxy.MergedReplay` trains on the exact frozen merged trajectories.
  It does not collect a second batch or use the baseline's updated weights.

`original_per_turn.py` is the unchanged splitter from the earlier ALFWorld
implementation. Only `split_per_turn` is called; its Session Server collection
wrapper is not used. Each split sample contains its complete preceding prefix,
and only the current completion contributes loss. Episode/group identity,
rewards, generated token IDs, log probabilities, and weight versions are kept.

Run the baseline first:

```bash
python scripts/run_qwen3_5_4b_swegym.py \
  --run-id proxy-per-turn-01 \
  --model-dir /source/models \
  --data-dir /workspace/turn-merge-ablation/data \
  --output-dir /workspace/proxy-turn-ablation/proxy-per-turn-01 \
  --public-url https://YOUR-MODAL-TUNNEL \
  --e2b-api-key-file /tmp/e2b/key \
  --num-rollout 1 --no-fully-async --no-save-optimizer \
  --extra-args '--rollout-function-path paired_proxy.Baseline --custom-rollout-log-function-path paired_proxy.log_rollout'
```

After it finishes, run the merged arm with a separate, initially empty output
directory, keeping `/source/models` unchanged:

```bash
python scripts/run_qwen3_5_4b_swegym.py \
  --run-id proxy-merge-01 \
  --model-dir /source/models \
  --data-dir /workspace/turn-merge-ablation/data \
  --output-dir /workspace/proxy-turn-ablation/proxy-merge-01 \
  --public-url https://YOUR-MODAL-TUNNEL \
  --e2b-api-key-file /tmp/e2b/key \
  --num-rollout 1 --no-fully-async --no-save-optimizer \
  --extra-args '--rollout-function-path paired_proxy.MergedReplay --custom-rollout-log-function-path paired_proxy.log_rollout'
```

These are the original experiment's paths and IDs. For a new study, choose new
run IDs/output directories and a fresh volume namespace. The preserved plugin
uses `/workspace/proxy-turn-ablation/paired-data`; if its manifest already
exists, the baseline replays it instead of recollecting. Do not reuse that
directory for a different model or dataset. Archive an existing study before
using the path for a new one. Both arms disable asynchronous prefetch because
this comparison contains exactly one shared rollout batch and one update each.

The manifest records the shared live rollout time, token/sample counts,
prefix duplication, conservation checks, and SHA-256 hashes. Read the live
rollout time from `ablation/shared_rollout_seconds`; the merged arm's native
rollout timer measures replay I/O. Training timers remain Miles' native timers.

The runtime modules intentionally retain the bytes used for the completed
experiment, including the fixed paired-data path and splitter's legacy
docstrings. Do not substitute a rewritten splitter when reproducing this pair.
Experiment logs, results, manifests, checkpoints, and credentials are stored
outside the Git repository.
