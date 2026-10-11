"""Train Qwen3.5-4B on ALFWorld through Miles Session Server v2, or through agent_proxy/.

Requires a Miles GPU environment, converted Qwen3.5-4B checkpoint, and indexed
ALFWorld games. See examples/experimental/alfworld/README.md for preparation.

Args:
  --model-dir / --data-dir: Model and dataset roots.
  --output-dir: Parent of run-specific checkpoints and rollout dumps.
  --num-rollout: Number of rollout/train iterations (default: two-step smoke run).
  --rollout-batch-size / --n-samples-per-prompt: Tasks and episodes per task.
  --num-gpus-per-node / --tensor-model-parallel-size: Single-node GPU layout.
  --max-steps / --max-response-len: Per-episode turns and per-call output tokens.
  --per-turn-samples: Split each episode into per-turn training samples (default: off).
  --use-proxy-server: Serve sessions with agent_proxy/ (default: off).
  --extra-args: Additional Miles train.py flags.

Example:
  python scripts/run_qwen3_5_4b_alfworld.py --num-rollout 2 --num-gpus-per-node 4
"""

import shlex
from dataclasses import dataclass, field

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 4
    tensor_model_parallel_size: int = 4
    num_rollout: int = 2
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    max_steps: int = 30
    max_response_len: int = 4096
    max_tokens_per_gpu: int = 24576
    per_turn_samples: bool = False
    use_proxy_server: bool = False
    extra_args: str = ""

    def __post_init__(self):
        if self.num_nodes != 1:
            raise ValueError("This initial ALFWorld recipe supports one node.")
        if (
            min(
                self.num_gpus_per_node,
                self.tensor_model_parallel_size,
                self.num_rollout,
                self.rollout_batch_size,
                self.max_steps,
                self.max_response_len,
                self.max_tokens_per_gpu,
            )
            < 1
        ):
            raise ValueError("GPU, batch, turn and token counts must be positive")
        if self.per_turn_samples and self.use_proxy_server:
            raise ValueError("--per-turn-samples splits Session Server v2 output; it does not apply to agent_proxy")
        if self.n_samples_per_prompt < 2:
            raise ValueError("GRPO needs at least two samples per prompt")
        if self.num_gpus_per_node % self.tensor_model_parallel_size:
            raise ValueError("num_gpus_per_node must be divisible by tensor_model_parallel_size")
        if self.global_batch_size % (self.num_gpus_per_node // self.tensor_model_parallel_size):
            raise ValueError("The episode batch must be divisible by data parallel size")

    @property
    def global_batch_size(self):
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def run_dir(self):
        return f"{self.output_dir}/alfworld/{self.run_id}"


def execute(args: ScriptArgs):
    if args.use_proxy_server:
        generate_path = "agent_proxy.miles_adapter.generate"
        # agent_proxy renders prompts from the checkpoint's own chat template with continuous token, so no
        # --tito-model. SGLang's reasoning/tool-call parsers are named explicitly (the qwen35 TITO values).
        session_args = (
            "--use-proxy-server --proxy-server-workers 1 "
            "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        )
    else:
        generate_path = (
            "examples.experimental.alfworld.per_turn.generate"
            if args.per_turn_samples
            else "miles.rollout.generate_hub.agentic_tool_call.generate"
        )
        session_args = "--use-session-server v2 --tito-model qwen35 --session-server-workers 1 "
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(args.model_dir + '/Qwen3.5-4B')} "
        f"--ref-load {shlex.quote(args.model_dir + '/Qwen3.5-4B_torch_dist')} "
        f"--load {shlex.quote(args.run_dir + '/checkpoints')} "
        f"--save {shlex.quote(args.run_dir + '/checkpoints')} --save-interval 20 "
    )
    rollout_args = (
        f"--prompt-data {shlex.quote(args.data_dir + '/alfworld/train.jsonl')} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        # The agent builds chat messages. Applying a template in the data loader
        # would render them before the session/TITO layer gets to see them.
        f"--custom-generate-function-path {generate_path} "
        "--custom-agent-function-path examples.experimental.alfworld.agent.run "
        "--custom-agent-function-mode subproc "
        "--custom-rm-path examples.experimental.alfworld.agent.require_reward "
        f"{session_args}"
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.max_response_len} --rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} --balance-data "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 1024 "
    )
    if args.tensor_model_parallel_size > 1:
        perf_args += "--sequence-parallel "
    algo_args = (
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 "
        "--kl-loss-type low_var_kl --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 10 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 "
    misc_args = (
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --colocate "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {shlex.quote(args.run_dir + '/dump_details')} "
    )
    args.create_backend().execute_train(
        train_args=" ".join(
            [
                ckpt_args,
                rollout_args,
                perf_args,
                algo_args,
                optimizer_args,
                sglang_args,
                misc_args,
                U.get_default_wandb_args(__file__, run_id=args.run_id),
                args.extra_args,
            ]
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-4B",
        megatron_path=args.megatron_path,
        extra_env_vars={"ALFWORLD_MAX_STEPS": str(args.max_steps)},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
