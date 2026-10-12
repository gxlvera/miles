"""Qwen3.5-4B GRPO through Claude Code, E2B and the the user agent_proxy server.

Prerequisites: prepared/calibrated templates, downloaded Qwen3.5-4B checkpoint,
E2B key file, and the authenticated gateway reachable from E2B over HTTPS.

Args:
  --prepare-only: Convert the downloaded Hugging Face checkpoint to Megatron.
  --model-dir / --data-dir / --output-dir: Persistent experiment paths.
  --public-url / --gateway-token-file: TLS gateway address and local token file.
  --num-rollout: Number of optimizer updates, one batch per rollout.
  --context-parallel-size / --max-tokens-per-gpu: Dynamic microbatch sizing.
  --actor-num-gpus-per-node / --rollout-num-gpus: Separate training/inference GPUs.
  --max-weight-staleness: Maximum oldest-token weight gap in fully async training.

Example:
  python scripts/run_qwen3_5_4b_swegym.py --model-dir /models --data-dir /data \
      --output-dir /results --public-url https://example.modal.host
"""

import os
import shlex
from dataclasses import dataclass, field

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    prepare_only: bool = False
    debug_rollout_only: bool = False
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/swegym"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    num_rollout: int = 40
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 16
    max_seq_len: int = 65536
    max_response_len: int = 16000
    max_turns: int = 80
    agent_timeout: int = 2400
    agent_concurrency: int = 16
    proxy_server_workers: int = 4
    save_interval: int = 5
    max_weight_staleness: int = 2
    fully_async: bool = True
    save_optimizer: bool = True
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int = 4
    context_parallel_size: int = 2
    max_tokens_per_gpu: int = 24576
    public_url: str = "https://example.modal.host"
    gateway_token_file: str = "/tmp/swegym-gateway-token"
    e2b_api_key_file: str = "/root/.config/e2b/api_key"
    extra_args: str = ""

    @property
    def tensor_parallel_size(self) -> int:
        return 2


def prepare(args: ScriptArgs):
    args.create_backend().convert_checkpoint(
        model_name="Qwen3.5-4B",
        megatron_model_type="qwen3.5-4B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/Qwen3.5-4B",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    quote = shlex.quote
    checkpoint = f"--hf-checkpoint {quote(args.model_dir + '/Qwen3.5-4B')} --ref-load {quote(args.model_dir + '/Qwen3.5-4B_torch_dist')} --load {quote(args.output_dir + '/checkpoints')} --save {quote(args.output_dir + '/checkpoints')} --save-interval {args.save_interval} "
    if not args.save_optimizer:
        checkpoint += "--no-save-optim "
    rollout = (
        f"--prompt-data {quote(args.data_dir + '/train.jsonl')} --input-key prompt --metadata-key metadata "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.max_response_len} "
        "--rollout-temperature 1.0 --rollout-shuffle --balance-data "
        "--custom-generate-function-path agent_proxy.miles_adapter.generate "
        "--custom-agent-function-path agent.run --custom-agent-function-mode inline --custom-rm-path agent.reward "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--use-proxy-server --proxy-server-workers {args.proxy_server_workers} --proxy-server-port 30000 "
        "--proxy-message-normalizer-path claude_history.normalize_message "
        f"--max-seq-len {args.max_seq_len} "
        "--session-sample-picker-path branch_policy.longest "
        "--session-sample-postprocessor-path branch_policy.postprocess --apply-chat-template-kwargs '{\"enable_thinking\":true}' "
    )
    performance = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.context_parallel_size} --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 256 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    algorithm = "--advantage-estimator grpo --use-tis --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 "
    optimizer = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    sglang = (
        f"--rollout-num-gpus-per-engine {args.tensor_parallel_size} --sglang-mem-fraction-static 0.65 "
        "--sglang-served-model-name Qwen3.5-4B --sglang-max-running-requests 8 "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 "
        "--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 "
        f"--sglang-context-length {args.max_seq_len} --sglang-router-port 31000 "
    )
    misc = (
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} --update-weights-interval 1 "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {quote(args.output_dir + '/traces')} "
        "--custom-rollout-log-function-path ablation_metrics.log_rollout "
    )
    if args.fully_async:
        misc += (
            f"--fully-async --max-weight-staleness {args.max_weight_staleness} "
            "--async-max-concurrent-samples 128 --async-data-buffer-capacity-factor 2 "
            "--async-unused-samples-handler retry "
        )
    if args.debug_rollout_only:
        misc += "--debug-rollout-only "
    env = {
        "PYTHONPATH": f"{U.repo_base_dir}/examples/experimental/swegym:{U.repo_base_dir}:{args.megatron_path}",
        "SWEGYM_DATA_DIR": args.data_dir,
        "SWEGYM_EXPERIMENT_ID": args.run_id,
        "SWEGYM_PROXY_WORKERS": str(args.proxy_server_workers),
        "SWEGYM_AGENT_CONCURRENCY": str(args.agent_concurrency),
        "SWEGYM_TRIALS_DIR": args.output_dir + "/trials",
        "SWEGYM_PROXY_PUBLIC_URL": args.public_url,
        "SWEGYM_GATEWAY_TOKEN_FILE": args.gateway_token_file,
        "SWEGYM_MAX_TURNS": str(args.max_turns),
        "SWEGYM_CONTEXT_LENGTH": str(args.max_seq_len),
        "SWEGYM_MAX_OUTPUT_TOKENS": str(args.max_response_len),
        "SWEGYM_AGENT_TIMEOUT": str(args.agent_timeout),
        "E2B_API_KEY_FILE": args.e2b_api_key_file,
    }
    args.create_backend().execute_train(
        train_args=checkpoint + rollout + performance + algorithm + optimizer + sglang + misc + _wandb_args(args) + " " + args.extra_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-4B",
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars=env,
        job_lifetime="launcher",
    )


def _wandb_args(args):
    # The shared helper places the key in argv. Remove it before command/log
    # construction; Ray inherits the Modal secret through the parent process.
    flags = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
    if not flags:
        return ""
    for name in ("--wandb-key", "--wandb-project", "--wandb-group"):
        index = flags.index(name)
        del flags[index : index + 2]
    flags += ["--wandb-project", os.environ.get("WANDB_PROJECT", "miles-swegym"), "--wandb-team", os.environ.get("WANDB_ENTITY", ""), "--wandb-group", args.run_id, "--wandb-run-id", args.run_id]
    return " " + shlex.join(flags) + " "


@U.dataclass_cli
def main(args: ScriptArgs):
    if args.prepare_only:
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    typer.run(main)
