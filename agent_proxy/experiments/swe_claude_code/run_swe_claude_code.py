"""GRPO for Qwen3.5-4B with Claude Code on SWE-Gym-Lite, served through agent_proxy.

One node, disaggregated: ``--train-gpus`` GPUs train (TP only; Qwen3.5's linear-attention layers
are not run with context parallelism here), ``--rollout-gpus`` GPUs serve SGLang. Each trajectory is
a Harbor ``claude-code`` trial in its own Modal sandbox; Claude Code reaches the proxy replica
through the public URL in ``AGENT_PROXY_PUBLIC_URLS`` (set by modal_app.py from Modal tunnels).

Needs: the Miles image with Harbor installed, the volume layout under ``--vol``
(models/Qwen3.5-4B, harbor_tasks/, data/), Modal credentials in the environment for the
sandboxes, and WANDB_API_KEY for online W&B logging.

Args:
  --run-name: W&B run name (exact) and checkpoint/dump directory name.
  --num-rollout / --rollout-batch-size / --n-samples-per-prompt: steps and episodes per step.
  --save-interval: checkpoint every N steps.
  --max-seq-len / --max-tokens-per-gpu: per-sample token cap and dynamic-batch budget.
  --debug-rollout-only: run rollouts without training (smoke test).

Example (inside the container):
  python agent_proxy/experiments/swe_claude_code/run_swe_claude_code.py --num-rollout 40
"""

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils as U

_HARBOR_EXAMPLE = Path(U.repo_base_dir) / "examples/experimental/harbor"
sys.path.insert(0, str(_HARBOR_EXAMPLE))
from launch_common import agentic_pythonpath_dirs, harbor_env_vars  # noqa: E402


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_name: str = "proxy-claude-code-swegym-qwen3.5-4b"
    vol: str = "/vol"
    model_name: str = "Qwen3.5-4B"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    train_gpus: int = 4
    rollout_gpus: int = 4
    tensor_model_parallel_size: int = 4
    prompt_data: str = "data/swegym_lite_train.jsonl"
    eval_prompt_data: str = "data/swebench_verified_eval32.jsonl"
    eval_interval: int = 0  # 0 disables SWE-bench Verified evaluation
    num_rollout: int = 40
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    save_interval: int = 5
    max_seq_len: int = 65536
    max_tokens_per_gpu: int = 32768
    max_output_tokens: int = 8192
    max_turns: int = 50
    agent_timeout: int = 3600
    proxy_server_workers: int = 4
    proxy_server_port: int = 30000
    debug_rollout_only: bool = False
    wandb_project: str = "humans&take-home"
    wandb_team: str = "xiaoleg-carnegie-mellon-university"
    extra_args: str = ""

    # read by launch_common.harbor_env_vars
    harbor_env_type: str = "modal"
    harbor_env_kwargs: str = ""
    harbor_tasks_dir: str = "/vol/harbor_tasks"
    harbor_trials_dir: str = "/tmp/harbor_trials"
    agent_model_name: str = "model"
    modal_config_file: str = ""

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def run_dir(self) -> str:
        return f"{self.vol}/runs/{self.run_name}"

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.vol}/models/{self.model_name}"


def prepare(args: ScriptArgs) -> None:
    """Convert the HF checkpoint to Megatron torch_dist once; later runs reuse it from the volume."""
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type="qwen3.5-4B",
        num_gpus_per_node=1,
        dir_dst=f"{args.vol}/models",
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def _harbor_env(args: ScriptArgs) -> dict[str, str]:
    # Read by harbor_agent_function on the rollout workers (forwarded by harbor_env_vars when set).
    os.environ.update(
        AGENT_MAX_OUTPUT_TOKENS=str(args.max_output_tokens),
        AGENT_MAX_INPUT_TOKENS=str(args.max_seq_len),
        HARBOR_AGENT_MAX_ITERATIONS=str(args.max_turns),
        HARBOR_MAX_SEQ_LEN=str(args.max_seq_len),
        AGENT_TRIAL_TIMEOUT=str(args.agent_timeout + 1800),
        HARBOR_TIMEOUT_MULTIPLIER="1.0",
    )
    return harbor_env_vars(args)


def execute(args: ScriptArgs) -> None:
    run_dir = args.run_dir
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.hf_checkpoint}_torch_dist "
        f"--load {run_dir}/checkpoints "
        f"--save {run_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
    )
    rollout_args = (
        f"--prompt-data {args.vol}/{args.prompt_data} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {args.max_output_tokens} "
        f"--max-seq-len {args.max_seq_len} "
        "--balance-data "
    )
    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-prompt-data swebench_verified {args.vol}/{args.eval_prompt_data} "
            f"--eval-interval {args.eval_interval} --n-samples-per-eval-prompt 1 "
        )
    agent_args = (
        "--custom-generate-function-path agent_proxy.miles_adapter.generate "
        "--custom-agent-function-path harbor_agent_function.run "
        "--custom-agent-function-mode subproc "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-proxy-server "
        f"--proxy-server-workers {args.proxy_server_workers} "
        f"--proxy-server-port {args.proxy_server_port} "
    )
    # TP only and full recompute. A trajectory longer than --max-tokens-per-gpu gets a micro-batch of its
    # own, so the peak is one --max-seq-len sample; TP4 keeps its vocab-parallel logits small.
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 1024 "
    )
    algo_args = (
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl "
        "--entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 --use-tis "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = (
        f"--rollout-num-gpus {args.rollout_gpus} --rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
    )
    misc_args = (
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {run_dir}/dump_details "
    )
    wandb_args = ""
    if os.environ.get("WANDB_API_KEY"):
        # The key stays in the environment; the run name is exactly --run-name.
        wandb_args = (
            f"--use-wandb --wandb-project {shlex.quote(args.wandb_project)} --wandb-team {args.wandb_team} "
            f"--wandb-group {args.run_name} --disable-wandb-random-suffix --wandb-dir {run_dir}/wandb "
        )
    debug_args = "--debug-rollout-only " if args.debug_rollout_only else ""

    extra_env_vars = {
        "PYTHONPATH": ":".join([args.megatron_path, *agentic_pythonpath_dirs(), str(U.repo_base_dir)]),
        "AGENT_PROXY_PUBLIC_URLS": os.environ.get("AGENT_PROXY_PUBLIC_URLS", "{}"),
        # Harbor passes this to Claude Code; the Anthropic route rejects requests with `thinking`.
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
        **_harbor_env(args),
    }
    args.create_backend().execute_train(
        train_args=" ".join(
            [
                ckpt_args,
                rollout_args,
                eval_args,
                agent_args,
                perf_args,
                algo_args,
                optimizer_args,
                sglang_args,
                misc_args,
                wandb_args,
                debug_args,
                args.extra_args,
            ]
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-4B",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
