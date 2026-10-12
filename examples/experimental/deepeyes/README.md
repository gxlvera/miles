# DeepEyes on agent_server

"Thinking with images" RL ([DeepEyes](https://arxiv.org/abs/2505.14362)), ported from verl's `recipe/deepeyes`. The policy answers a question about an image and may call `image_zoom_in_tool` to look at a crop, for at most 5 assistant turns. Every crop is a new image in the conversation, which makes this a test of the multimodal path in [agent_server](../../../agent_server/DESIGN.md#6-multimodal).

- `agent.py`: the harness, a Miles agent function. It talks OpenAI chat to the agent server URL Miles passes in, sends each image as a PNG data URI, and runs the tool itself by cropping the original image. The tool reads `bbox_2d` on Qwen3-VL's relative 0–1000 grid, not in pixels as the Qwen2.5-VL recipe did; `probe_bbox.py` checks which frame a policy uses.
- `reward.py`: the recipe's reward, 0.8 × accuracy + 0.2 × format + 1.2 × (tool used and correct). Accuracy comes from an LLM judge when `DEEPEYES_JUDGE_URL` points to an OpenAI-compatible server (`DEEPEYES_JUDGE_MODEL` names its model). Otherwise a lenient word match against the reference answer is used, which needs no extra GPUs.
- `prepare_data.py`: samples 4,000 rows of the V* visual-search part of `ChenShawn/DeepEyes-Datasets-47k` (22,362 rows) and writes `train.jsonl` plus PNG images, downscaled to at most 1M pixels. Miles draws training prompts without replacement within an epoch, so a run sees no prompt twice while it draws fewer than 4,000. A 30-step run draws about 900.
- `prepare_vstar.py`: writes V* Bench (191 multiple-choice questions on high-resolution images) as the eval set. Images keep their full resolution. The harness shows the model a view of at most 1M pixels and crops zoom-ins from the original. Its reward on these rows is plain accuracy.
- `run_qwen3_vl_4b_deepeyes.py`: Qwen3-VL-4B-Instruct on one node: 2 GPUs train, 2 GPUs serve rollouts, fully async with weights at most one version stale. It evaluates V* Bench accuracy before training and every 10 steps, and saves the final weights (HF export) under `<vol>/runs/<run-name>/`.

Compared with the recipe: the policy is Qwen3-VL-4B-Instruct instead of Qwen2.5-VL-7B, training uses a 4,000-row sample of the V* part, the async runs stay at most one step off-policy (corrected with TIS), and the default judge is rule-based.

SGLang needs [agent_server/sglang_patch](../../../agent_server/sglang_patch/README.md). On Modal, [agent_server/experiments/deepeyes/modal_app.py](../../../agent_server/experiments/deepeyes/modal_app.py) builds that image, prepares the data, and runs the training.

```bash
python examples/experimental/deepeyes/prepare_data.py --out /vol/data/deepeyes
python examples/experimental/deepeyes/prepare_vstar.py --out /vol/data/vstar_bench
python examples/experimental/deepeyes/run_qwen3_vl_4b_deepeyes.py --vol /vol --num-rollout 30
```
