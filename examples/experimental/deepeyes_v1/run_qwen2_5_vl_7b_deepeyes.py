"""DeepEyes v1 GRPO for Qwen2.5-VL-7B-Instruct (seed-verl's scripts/deepeyes_v1 recipe), served through agent_server.

One node, disaggregated and fully async: ``--train-gpus`` GPUs train (Megatron through the HF bridge,
TP only), ``--rollout-gpus`` GPUs serve SGLang, and rollout keeps generating while training runs, using
weights at most ``--max-weight-staleness`` versions old. Each episode is the harness in agent.py talking
OpenAI chat to an agent server replica, with the recipe's chat template and without a tool-call parser.

Recipe settings kept: GRPO with 8 samples per prompt, lr 1e-6, no KL, at most 5 assistant turns,
prompts up to 8192 tokens and 16384 response tokens per episode, temperature 1.0, greedy validation.

Needs: the Miles image with agent_server/sglang_patch applied to SGLang; the HF checkpoint under
``<vol>/models``; the data from prepare_data.py under ``<vol>/data/deepeyes_v1`` and V* Bench from
examples/experimental/deepeyes/prepare_vstar.py under ``<vol>/data/vstar_bench``; the judge
(Qwen2.5-7B-Instruct on SGLang, see reward.py) at DEEPEYES_JUDGE_URL on a GPU outside
``--num-gpus-per-node``; WANDB_API_KEY for online W&B logging.

Args:
  --run-name: W&B run name (exact).
  --num-rollout / --rollout-batch-size / --n-samples-per-prompt: steps and episodes per step.
  --max-response-len: tokens per assistant turn; --max-seq-len: per-episode token cap.
  --max-weight-staleness: how many weight versions rollout may lag behind training.
  --eval-interval: evaluation every N steps and before the first: the reward on the 512-row held-out
    split and V* Bench accuracy (greedy, one sample per question).
  --save-interval: checkpoint every N steps; the last step is always saved. Each save writes a
    Megatron checkpoint without optimizer state and an HF export under ``<vol>/runs/<run-name>/``.

Example (inside the container, judge already serving):
  python examples/experimental/deepeyes_v1/run_qwen2_5_vl_7b_deepeyes.py --num-rollout 30
"""

import os
import shlex
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_name: str = "agent-server-deepeyes-v1-qwen2.5-vl-7b"
    vol: str = "/vol"
    model_name: str = "Qwen2.5-VL-7B-Instruct"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 7  # the node's eighth GPU serves the judge
    train_gpus: int = 4
    rollout_gpus: int = 3
    tensor_model_parallel_size: int = 2
    prompt_data: str = "data/deepeyes_v1/train.jsonl"
    val_prompt_data: str = "data/deepeyes_v1/val.jsonl"
    vstar_prompt_data: str = "data/vstar_bench/test.jsonl"
    eval_interval: int = 10  # also before the first step; 0 disables evaluation
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    max_response_len: int = 8192
    max_seq_len: int = 8192 + 16384
    # Samples packed per micro-batch; a longer sample trains alone. The vision tower is not recomputed.
    max_tokens_per_gpu: int = 12288
    max_weight_staleness: int = 1
    save_interval: int | None = 10
    agent_server_workers: int = 2
    wandb_project: str = "humans&take-home"
    wandb_team: str = "xiaoleg-carnegie-mellon-university"
    extra_args: str = ""

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.vol}/models/{self.model_name}"

    @property
    def run_dir(self) -> str:
        return f"{self.vol}/runs/{self.run_name}"


def execute(args: ScriptArgs) -> None:
    example = f"{U.repo_base_dir}/examples/experimental/deepeyes_v1"
    # The HF checkpoint is loaded through the Megatron bridge; no torch_dist conversion and no reference
    # model (the recipe trains without KL).
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --load {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        # Model weights only (no optimizer state): the HF export is what later evaluation or serving loads.
        f"--save {args.run_dir}/checkpoints --save-interval {args.save_interval or args.num_rollout} --no-save-optim "
        f"--save-hf {args.run_dir}/hf/iter_{{rollout_id}} "
    )
    rollout_args = (
        f"--prompt-data {args.vol}/{args.prompt_data} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {args.max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        "--balance-data "
        f"--fully-async --max-weight-staleness {args.max_weight_staleness} "
    )
    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-prompt-data deepeyes_val {args.vol}/{args.val_prompt_data} "
            f"vstar_bench {args.vol}/{args.vstar_prompt_data} --eval-interval {args.eval_interval} "
            f"--n-samples-per-eval-prompt 1 --eval-temperature 0 --eval-max-response-len {args.max_response_len} "
            "--custom-eval-rollout-log-function-path examples.experimental.deepeyes_v1.agent.log_eval "
        )
    agent_args = (
        "--custom-generate-function-path agent_server.miles_adapter.generate "
        "--custom-agent-function-path examples.experimental.deepeyes_v1.agent.run "
        "--custom-agent-function-mode inline "
        "--custom-rm-path examples.experimental.deepeyes_v1.agent.score_failed_episodes "
        "--custom-rollout-log-function-path examples.experimental.deepeyes_v1.agent.log_rollout "
        # A group with an episode that produced no turn (its first request failed) is not trained on.
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--use-agent-server --agent-server-workers {args.agent_server_workers} "
        "--use-miles-router "
        # The recipe's template renders the tool schema and the tool turns; Miles hands it to SGLang too.
        f"--chat-template-path {example}/chat_template.jinja "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 1024 "
    )
    # The recipe on verl's actor defaults: GRPO (std-normalized), no KL, no entropy bonus, clip 0.2 both
    # ways, token-mean loss, AdamW lr 1e-6 constant, weight decay 0.01, betas (0.9, 0.999), grad clip 1.0.
    # TIS corrects for the one step of staleness.
    algo_args = "--advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --calculate-per-token-loss --use-tis "
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 "
        "--adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0 "
    )
    # No tool-call parser: the harness parses the raw replies the way the recipe does.
    sglang_args = (
        f"--rollout-num-gpus {args.rollout_gpus} --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "
    )
    misc_args = (
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    wandb_args = ""
    if os.environ.get("WANDB_API_KEY"):
        wandb_args = (
            f"--use-wandb --wandb-project {shlex.quote(args.wandb_project)} --wandb-team {args.wandb_team} "
            f"--wandb-group {args.run_name} --disable-wandb-random-suffix "
            f"--wandb-dir {args.run_dir}/wandb "
        )

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
                args.extra_args,
            ]
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        # Qwen2.5-VL-7B's language model has the shape of Qwen2.5-7B.
        megatron_model_type="qwen2.5-7B",
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTHONPATH": f"{args.megatron_path}:{U.repo_base_dir}"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
