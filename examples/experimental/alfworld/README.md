# ALFWorld through Session Server v2

Train a small whitebox tool agent with Qwen3.5-4B, using Miles' existing TITO and multi-turn trajectory collection.

The default mode uses native merged samples. An optional per-turn baseline splits
them before training. The Session Server, sample postprocessor and trainer are
unchanged. No Harbor, Docker tool sandbox,
history compaction, branching or custom Trajectory Manager is involved.

```text
Miles dataset (one game per prompt)
  -> agentic_tool_call.generate (one session per episode)
  -> agent.run (TextWorld env + OpenAI tool-call loop, in a subprocess)
  -> Session Server v2 / qwen35 TITO -> SGLang
  -> native merged samples + terminal reward -> GRPO trainer
```

## Prepare on the training machine

Use a working **Linux Miles GPU environment** with the versions of SGLang,
Megatron and Transformers required by this checkout. Run commands from the Miles
repository root, with Miles installed (`pip install -e .` in that environment).
The default launcher uses one node with **4 GPUs**, Megatron TP=4, colocated
sampling, and one GPU per SGLang engine. This layout is a starting configuration;
GPU memory requirements have not yet been measured for this example.

Install the text environment (no THOR/visual extras):

```bash
pip install -r examples/experimental/alfworld/requirements.txt
```

Only the official compiled TextWorld games are needed. This small download avoids
the visual detector weights bundled by `alfworld-download`:

```bash
mkdir -p /root/datasets/alfworld/raw
curl -fL https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip \
  -o /root/datasets/alfworld/games.zip
unzip -q /root/datasets/alfworld/games.zip -d /root/datasets/alfworld/raw

python -m examples.experimental.alfworld.prepare_data \
  --data-dir /root/datasets/alfworld/raw \
  --output /root/datasets/alfworld/train.jsonl \
  --split train --check-env
```

`--check-env` resets and steps the first game on CPU before spending GPU time.
An existing `alfworld-download` directory also works as `--data-dir`. The index
includes only supported games marked solvable, and stores **absolute game paths**;
generate it on the training machine, or use paths mounted identically there.
`--limit 16` can shorten dataset preparation for an initial check. It is a subset,
not a different task or reward. Evaluation splits in the official archive are
`valid_seen` and `valid_unseen`; this first launcher does not run evaluation.

Download the model and convert it with Miles' own model definition. The following
conversion command uses **Bash** arrays (run this block in Bash):

```bash
hf download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B
MODEL_ARGS_LINE="$(python miles/utils/external_utils/model_args_utils.py qwen3.5-4B)" || exit 1
read -ra MODEL_ARGS <<< "$MODEL_ARGS_LINE"
PYTHONPATH="$PWD:/root/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}" \
  python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/models/Qwen3.5-4B \
  --save /root/models/Qwen3.5-4B_torch_dist
```

Skip conversion if that directory already contains a checkpoint converted by this
Miles version. The model directory must contain both the HF and `_torch_dist`
directories; the launcher takes their parent as `--model-dir`.

## Run two rollout/train iterations

```bash
python scripts/run_qwen3_5_4b_alfworld.py \
  --model-dir /root/models \
  --data-dir /root/datasets \
  --output-dir /root/shared_data \
  --run-id alfworld-smoke-01 \
  --num-gpus-per-node 4 \
  --num-rollout 2
```

The launcher uses the standard Miles `command_utils` backend, including its Ray
startup/cleanup; use a dedicated training allocation. For an already-running Ray
cluster, set `MILES_SCRIPT_EXTERNAL_RAY=1` as in other Miles launchers.

The launcher explicitly selects the reasoning and tool-call parsers bound to
Miles' `qwen35` TITO definition. Template auto-detection alone is insufficient
in some SGLang images: without an active tool-call parser, valid tool syntax can
arrive as plain text and every environment action will be rejected.

Defaults: 2 tasks × 4 episodes = 8 episodes per rollout and one GRPO batch per
iteration; 30 agent turns; up to 4096 generated tokens **per model call**. An
invalid action consumes a turn but does not advance TextWorld. Reward is 1 only
when `infos['won']` is true, otherwise 0 on normal episode termination.
Environment/model-request exceptions are not converted to reward 0.

For the original recipe's batch scale, use `--rollout-batch-size 16
--n-samples-per-prompt 8`; the global batch size is derived automatically.
`--max-steps`, `--max-response-len`, `--max-tokens-per-gpu` and
`--tensor-model-parallel-size` are exposed for resource adjustments.
The batching token budget does **not** truncate a long individual trajectory.

For one-batch-ahead asynchronous training on eight GPUs, add `--async-train
--num-gpus-per-node 8 --tensor-model-parallel-size 4 --rollout-batch-size 4
--n-samples-per-prompt 8`. The launcher splits GPUs equally between trainer
and rollout pools, selects `train_async.py`, uses rollout behavior log-probs,
and performs one optimizer step and one weight publication per batch. The
driver waits for the single prefetched batch before publishing weights, so
trajectories are at most one optimizer update behind and never change weights
mid-episode. This mode does not enable the persistent fully-async producer.

This example preserves full histories; reduce turn/output limits for a smaller
memory smoke run, and keep those settings identical in later representation
comparisons. It does not yet demonstrate 100-turn / 128K-context training.

## What to check

- Both iterations finish rollout collection and optimizer work without agent or
  TITO errors. A two-step run checks execution, not convergence; a batch where all
  episodes have equal reward has zero GRPO reward advantage.
- Native rollout dumps and trajectories are under
  `<output-dir>/alfworld/<run-id>/dump_details/`. Check `task_success`, `num_turns`,
  `num_steps`, `invalid_action_count`, `stop_reason`, reward and loss masks.
  The ordinary append-only dialogue should become one merged training sample per
  episode, with environment tokens masked out by Miles.
- Use Miles' training timer for timing comparisons, separately from rollout
  and startup time; see the per-turn baseline below.

## Optional token diagnostics for merged samples

Add these flags through `--extra-args` to retain the ordinary Miles metrics and
also log per-turn token lengths to W&B, stdout and `token_metrics/` next to
`dump_details/`:

```text
--custom-rollout-log-function-path examples.experimental.alfworld.token_metrics.log_rollout
--custom-eval-rollout-log-function-path examples.experimental.alfworld.token_metrics.log_eval
```

The diagnostics read native token IDs and the loss mask; they do not split or
re-tokenize training samples. Assistant counts are completion-mask spans.
Non-assistant counts include the first prompt and subsequent tool/context
tokens, reported both as newly added tokens and accumulated non-assistant
context. Means weight each assistant turn equally. Eval publishes only these
cross-turn means and episode aggregates; training rollout diagnostics also
include per-position means. Raw per-episode turn counts remain in the local
diagnostic JSON for inspection. A final environment reply after the last completion is not a model
input and is absent from these counts.

## Per-turn training baseline

After the same preparation, run each mode with a **different fresh run ID**:

```bash
python scripts/run_qwen3_5_4b_alfworld.py \
  --model-dir /root/models --data-dir /root/datasets \
  --output-dir /root/shared_data --num-gpus-per-node 4 \
  --per-turn-samples --run-id alfworld-per-turn-01 --num-rollout 2

python scripts/run_qwen3_5_4b_alfworld.py \
  --model-dir /root/models --data-dir /root/datasets \
  --output-dir /root/shared_data --num-gpus-per-node 4 \
  --no-per-turn-samples --run-id alfworld-merged-01 --num-rollout 2
```

`--per-turn-samples` is **off by default**: omit it (or use
`--no-per-turn-samples`) to keep native merged samples. Enabling it changes only
the generate adapter. `per_turn.generate` calls the
native agentic adapter, then splits its merged output at the completion spans in
the loss mask. For this append-only harness, ones are model outputs and zeros
are the intervening tool/context tokens. The number of spans must match the
agent's reported turn count. Each turn receives its **full preceding history**,
but only its own output has loss. Original token IDs, rollout logprobs and
weight-version spans are preserved; decoding is for display, never re-tokenizing.

Miles already supports this:

- `GenerateFnOutput.samples` accepts a list. One task prompt still generates
  `n_samples_per_prompt` **episodes**, and each episode now returns multiple turns.
- All turns retain the same `group_index`, `index`, terminal reward and shared
  `rollout_id`. Native reward processing normalizes once per episode within the
  prompt group and broadcasts that advantage to its turns.
- Native `rollout_mask_sums` normalizes loss by the episode's total trainable
  tokens. The Megatron rollout-side scheduler batches by `rollout_id`, so the
  default global batch remains **8 episodes and one optimizer step per rollout**,
  regardless of the number of physical training samples. Do not multiply the
  batch size by the number of turns or add `--use-dynamic-global-batch-size`.

Check the logs for `Rollout-side DP schedule`: both runs should show
`num_rollouts=[8]`, with one entry in `num_microbatches`. The per-turn run has
more `num_samples` and may need more microbatches within that same optimizer step.
Each episode also logs `merged_tokens`, `per_turn_tokens` and `trainable_tokens`.
A three-turn CPU fixture has sequence lengths 5, 11 and 18: per-turn processing
reads 34 tokens versus 18 merged, while both train on the same 9 output tokens.
This token ratio is not a predicted GPU speedup.

Compare **`perf/actor_train_time`** (actor forward/backward and optimizer) and
**`perf/train_time`** (also includes reference/old-policy logprob computation and
advantages). Exclude rollout, initialization and checkpointing from this training
comparison. The first iteration includes warm-up; two iterations are a smoke
measurement, not a stable benchmark. Keep GPU layout, batch, turn/output limits
and token budget identical. Independent sampling can produce different lengths;
a strict comparison should later replay identical trajectories in both layouts.
`--load-debug-rollout-data` bypasses generation, so changing `--per-turn-samples` alone
does **not** split an existing dump; it must already contain the desired layout.

This measures duplicated history computation **in training**. Session Server
still assembles merged samples first; it is not an unmerged-storage benchmark.

For batch-level comparison metrics in either mode, set
`--custom-rollout-log-function-path examples.experimental.alfworld.benchmark_metrics.log_rollout`.
This replaces the merged-only token diagnostics hook. It logs `benchmark/*`
metrics to W&B and writes one JSON per rollout under `benchmark_metrics/` next
to `dump_details/`. `samples_per_source_prompt` divides by distinct prompt
groups; `samples_per_episode` divides by independently sampled trajectories.
Token totals count full unpadded sample sequences before the causal shift.
Repeated-history tokens are the sum of sample lengths minus the final merged
sequence length for each episode. Identical text in separate episodes is not
deduplicated, and lexical repetition is not counted as repeated history.
The hook preserves native training timings; sampling time is logged separately
as `benchmark/rollout_seconds`. Report optimizer/actor time, total train time,
and end-to-end step time separately because asynchronous work overlaps.

The splitter targets this dense, linear ALFWorld recipe: no compaction, branching,
R3, OPD, top-k/support replay or `--max-seq-len` truncation. Per-call output-length
termination is supported. Episode-level environment/cache totals are counted
once; native row-averaged rollout statistics (such as reward) may still be weighted
by the number of turns and should not be used as an episode success-rate comparison.

## Provenance and validation

The loop and hyperparameters are based on the Apache-2.0
[Dressage ALFWorld whitebox recipe](https://github.com/Accio-Lab/Dressage/tree/main/dressage/recipes/alfworld)
and its Qwen3.5-4B launcher: GRPO, LR 1e-6, KL coefficient 0.001, weight decay
0.01, temperature 1, 30 turns and 4096 output tokens per call. The integration
uses Miles' `qwen35` parser/template contract, structured tool calls, ALFWorld's
official readable-name wrapper, and a smaller default batch. It is an adaptation,
not a claim to reproduce Dressage's learning curve.

Local checks cover multi-turn HTTP payloads, unchanged assistant history,
invalid/multiple tool calls, terminal/length/turn stopping, environment cleanup,
missing rewards, native v2 reward postprocessing, dataset indexing and the recorded
Miles launch command. Baseline CPU tests additionally exercise native merge/split
round trips, unchanged training tokens/logprobs, unequal episode lengths, native
GRPO normalization, loss-reducer value/gradient equivalence on a toy token loss,
and one-step scheduling at DP=1 and DP=2. These are CPU checks, not GPU numerical
equivalence or a timing result. An official
downloaded game was completed in 5 turns with the real TextWorld environment and
a scripted HTTP model stub, producing reward 1. This checks the environment loop,
not model quality or actual Session Server/SGLang execution. GPU sampling/training
and reward improvement remain untested.

17 focused CPU tests and 2 launcher checks passed. The complete launch-script
suite currently cannot load its root fixtures because this CPU environment lacks
SGLang. An earlier isolated run of the broader launcher tests also did not pass: some
tests require Linux `/proc` or writable `/root`, and the checkout also has unrelated
missing/stale launcher snapshots. No shared test harness or existing snapshots
were changed to accommodate this example.

Focused CPU tests (the `--confcutdir` avoids loading unrelated GPU-stack fixtures):

```bash
pytest --confcutdir=tests/fast/alfworld tests/fast/alfworld
pytest --confcutdir=tests/manual/launch_scripts \
  tests/manual/launch_scripts/test_py_launch_scripts.py -k alfworld
```
