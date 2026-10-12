# τ²-bench on agent_server (Qwen3-4B-Instruct-2507)

Customer-service RL on [τ²-bench](https://github.com/sierra-research/tau2-bench), retail and airline. Each episode works like this:
- The policy talks to a simulated customer, an LLM that knows the task's scenario.
- The policy calls the domain's tools against a database.
- The reward is τ²-bench's own check.

So every episode is a long conversation in which user messages and tool results alternate, which exercises the agent server's continuous token on both kinds of append.

- `agent.py` is the harness, a Miles agent function. τ²-bench runs the episode through the async fork AReaL trains with (`dhh1995/tau2-bench@dhh/async-and-custom-completion`).
  - The policy is tau2's `LLMAgent`; its completions go to the agent server URL Miles passes in.
  - The customer is tau2's `UserSimulator`, served separately (`TAU2_USER_URL`) at temperature 0, as in τ²-bench.
  - The episode opens with τ²-bench's fixed greeting and ends when either side stops, or after `TAU2_MAX_STEPS` messages (100).
  - The reward is `evaluate_simulation(EvaluationType.ALL)`: the database state after the episode against the one the expected actions produce, times whether the agent communicated what it had to. It runs in a process pool, because it rebuilds and hashes whole databases.
  - Each training task runs on its own database variant. The variant is registered with tau2 as a domain of its own, so the evaluation replays the expected actions on the same database.
  - `log_rollout` and `log_eval` add database matches, how episodes ended and turn counts to W&B, and the eval reward per domain.
- `prepare_data.py` writes the data. Every task is validated against tau2's `Task` model.
  - Training: the RL split of [AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data), retail 563 and airline 1,148 synthetic tasks, each with its own database variant.
  - Evaluation: the official τ²-bench test split, retail 40 and airline 20, on the domains' own databases.
- `run_qwen3_4b_tau2.py`: Qwen3-4B-Instruct-2507 on one 8-GPU node. 2 GPUs train (Megatron via the HF bridge, TP 2) and 4 serve the policy; the remaining 2 serve the user simulator (Qwen3-30B-A3B-Instruct-2507, TP 2). It runs fully async, with weights at most one version stale.

Policy model. Miles' τ-bench example uses Qwen3-4B-Instruct-2507, without thinking; AReaL's τ² example uses the Qwen3 hybrid models with thinking on. This recipe takes the instruct model. Its chat template only appends, so the agent server's prompts equal a full render of the conversation. Qwen3 thinking templates drop earlier thinking once a new user message arrives. They need Miles' fixed templates (`qwen3_fixed.jinja` with `clear_thinking: false`) for that.

Settings:
- From Miles' τ-bench example: GRPO with 8 samples per prompt, lr 1e-6, no KL, clip 0.2/0.28, replies of up to 1024 tokens, temperature 1.
- From τ²-bench's defaults: evaluation at temperature 0, with one trial per task. The mean reward is pass^1.
- Data: 16 prompts per step, so that a 30-step run draws fewer than the 1,711 training tasks and repeats none.
- Context: an episode is capped at 32,768 tokens.

SGLang needs [agent_server/sglang_patch](../../../agent_server/sglang_patch/README.md). On Modal, [agent_server/experiments/tau2/modal_app.py](../../../agent_server/experiments/tau2/modal_app.py) builds that image with tau2, prepares the data, starts the user simulator and runs the training.

```bash
TAU2_DATA_DIR=/root/tau2-bench/data python examples/experimental/tau2/prepare_data.py --out /vol/data/tau2
CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server --model-path /vol/models/Qwen3-30B-A3B-Instruct-2507 \
    --served-model-name user --tp 2 --port 18902 --sampling-defaults openai  # after the launcher has started Ray
TAU2_DATA_DIR=/root/tau2-bench/data python examples/experimental/tau2/run_qwen3_4b_tau2.py --vol /vol --num-rollout 30
```
