# Claude Code + SWE-Gym on E2B through agent_proxy

This recipe uses the user's `gxlvera/miles` Proxy implementation at `fb93811e`,
with Qwen3.5-4B and the E2B task templates calibrated for this experiment.
It runs `train_async.py` with four training and four inference H100 GPUs.
The original HF chat template and Proxy continuous-token builder construct
trajectories; the Session Server is not started.

Prepare `train.jsonl`, `tasks.json`, and `templates.json` under `--data-dir`
using `prepare.py`. The verifier uses a fresh sandbox and only the candidate
patch; the agent does not receive gold test patches. Template creation and
verification follow `sandbox.py` and `templates.py`.

Run the authenticated `gateway.py` on port 30080 behind a Modal HTTPS tunnel.
It forwards only Anthropic messages and token counting, with replica routing
at ports 30000–30003. Then launch:

```bash
python scripts/run_qwen3_5_4b_swegym.py \
  --model-dir /source/models --data-dir /workspace/turn-merge-ablation/data \
  --output-dir /workspace/turn-merge-ablation/merged-proxy-40-01 \
  --public-url https://YOUR-MODAL-TUNNEL \
  --e2b-api-key-file /tmp/e2b/key --num-rollout 40
```

The launcher owns defaults: 64 episodes/update, 64K context, TP2/CP2 training,
dynamic token budget 24,576/GPU, full recompute, optimizer CPU offload,
GRPO + TIS, and a maximum oldest-token weight gap of 2. Checkpoints include
optimizer state and are saved every five updates. `--load` points to the same
checkpoint directory, so a restart resumes the last saved update.

When a trajectory has multiple leaves, `branch_policy.longest` picks the
longest native Proxy sample and records the discarded branch counts. The
Proxy owns trajectory assembly; the recipe does not reimplement it.

Two observed Claude compatibility fixes are recorded in the experiment trace:

- Tool arguments from the OpenAI protocol are normalized with Miles' existing
  JSON-to-dict helper before rendering the HF template.
- An optional `--proxy-message-normalizer-path` affects history identity only.
  This recipe explicitly selects the Qwen3.5 text-trimming projection; token
  equality tests verify the observed whitespace-only Claude round trip. Tool
  argument values and original generated tokens are preserved.

CPU checks use the real local Qwen3.5 tokenizer:

```bash
AGENT_PROXY_TEST_TOKENIZER=/source/models/Qwen3.5-4B \
  python -m pytest --confcutdir=agent_proxy agent_proxy/tests \
  tests/fast/examples/experimental/swegym
```

The earlier per-turn/merged ablation is a separate frozen source snapshot and
does not include these changes. Its paired timings should not be compared as
a benchmark of the new Proxy implementation.
