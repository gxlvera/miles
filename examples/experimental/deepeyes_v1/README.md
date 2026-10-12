# DeepEyes v1 on agent_server (Qwen2.5-VL-7B)

seed-verl's `scripts/deepeyes_v1` recipe, the upstream verl `recipe/deepeyes` (DeepEyes v1: a single zoom tool, no cold-start SFT, base Qwen2.5-VL-7B-Instruct), run on Miles through [agent_server](../../../agent_server/DESIGN.md). The policy answers a question about an image and may call `image_zoom_in_tool` to look at a crop, for at most 5 assistant turns.

- `agent.py`: the harness, a Miles agent function.
  - The prompt is the dataset's system and user messages. The tool schema is rendered by the recipe's chat template.
  - Replies are parsed like verl's `DeepEyesToolParser` on the raw text: an `<answer>` ends the episode, and only the last `<tool_call>` runs.
  - The tool follows `DeepEyesZoomInTool`. Boxes are in pixels of the image the model sees (both pass through `qwen_vl_utils.fetch_image`). Each box is clamped and rejected when a side is ≤ 30 px or the aspect ratio is > 100. The crop is snapped to the 28 px grid and followed by the official `USER_PROMPT_V2`.
  - SGLang runs without a tool-call parser, so each reply is replayed exactly as generated.
  - `log_rollout`/`log_eval` add judge accuracy, format errors and tool use to W&B, and the held-out reward per data source.
- `reward.py`: the recipe's reward, 0.8 × accuracy − 0.2 × format error + 1.2 × (tool used and correct). The range is −0.2 to 2.0. Accuracy is the verdict of an LLM judge (Qwen2.5-7B-Instruct on SGLang, `DEEPEYES_JUDGE_URL`) with the official judge prompt. V* Bench rows score plain multiple-choice accuracy instead.
- `chat_template.jinja`: the recipe's `custom_chat_template`, verbatim.
- `prepare_data.py`: the recipe's split.
  - V* + ThinkLite from DeepEyes-Datasets-47k (Chart left out), shuffled with seed 42.
  - The first 512 rows form the validation set; the rest are training rows.
  - Prompts over 8192 tokens are dropped, as verl's `filter_overlong_prompts` does.
  - Only the first 4,000 training rows are written. They are a random sample, and a 30-step run draws 960 of them, none twice.
- `run_qwen2_5_vl_7b_deepeyes.py`: one 8-GPU node.
  - 4 GPUs train (Megatron via the HF bridge, TP 2), 3 serve rollouts, and the eighth serves the judge.
  - Fully async, with weights at most one version stale.
  - Before training and every 10 steps it evaluates the held-out split (the recipe's validation reward) and V* Bench accuracy.
  - It saves a checkpoint (Megatron without optimizer state, plus an HF export) every 10 steps.

Kept from the recipe:
- GRPO, 8 samples per prompt, no KL;
- lr 1e-6 on verl's actor defaults (clip 0.2, token-mean loss, weight decay 0.01);
- prompts up to 8192 tokens and 16,384 response tokens per episode;
- temperature 1.0, greedy validation.

Differences:
- Rollouts are served by SGLang behind the agent server, not vLLM in verl's agent loop.
- Training uses Megatron instead of FSDP.
- Rollout runs up to one step ahead of training, corrected with TIS, where verl is on-policy.
- 32 prompts per step, the batch of the recipe's 2-GPU launcher; the 7-GPU launcher uses 112.
- A turn is capped at 8192 new tokens.

SGLang needs [agent_server/sglang_patch](../../../agent_server/sglang_patch/README.md). On Modal, [agent_server/experiments/deepeyes_v1/modal_app.py](../../../agent_server/experiments/deepeyes_v1/modal_app.py) builds that image, prepares the data, starts the judge and runs the training.

```bash
python examples/experimental/deepeyes_v1/prepare_data.py --out /vol/data/deepeyes_v1 --processor /vol/models/Qwen2.5-VL-7B-Instruct
python examples/experimental/deepeyes/prepare_vstar.py --out /vol/data/vstar_bench
CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server --model-path /vol/models/Qwen2.5-7B-Instruct \
    --served-model-name judge --port 18901 --context-length 4096  # after the launcher has started Ray
python examples/experimental/deepeyes_v1/run_qwen2_5_vl_7b_deepeyes.py --vol /vol --num-rollout 30
```
