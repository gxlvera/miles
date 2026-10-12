"""Modal app for the DeepEyes v1 run (examples/experimental/deepeyes_v1: seed-verl's recipe, Qwen2.5-VL-7B).

The image is the Miles image with agent_server/sglang_patch applied to its SGLang. One 8-GPU node
(H200, else H100): 4 GPUs train, 3 serve rollouts, the eighth serves the judge (Qwen2.5-7B-Instruct on SGLang).

- prepare: the policy and judge checkpoints, the v1 train/val split and V* Bench on the volume (CPU);
- check: turn-by-turn prompts of real episodes against the one-shot HF processor, and the tests (CPU);
- smoke: a 3-step run on small eval sets, with an evaluation and a checkpoint save;
- train: the run.

Usage (from the miles repo root, with a Modal CLI >= 1.6):
    modal run agent_server/experiments/deepeyes_v1/modal_app.py::prepare
    modal run agent_server/experiments/deepeyes_v1/modal_app.py::check
    modal run agent_server/experiments/deepeyes_v1/modal_app.py::smoke
    modal run --detach agent_server/experiments/deepeyes_v1/modal_app.py::train --launcher-args "--num-rollout 30"
"""

import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import modal

MILES_IMAGE = "radixark/miles:dev-202610102218"
POLICY = "Qwen2.5-VL-7B-Instruct"
JUDGE = "Qwen2.5-7B-Instruct"
JUDGE_PORT = 18901
# Only meaningful on the machine that launches the app; inside the container this file is /root/modal_app.py.
LOCAL_MILES = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/miles_src")
LAUNCHER = "examples/experimental/deepeyes_v1/run_qwen2_5_vl_7b_deepeyes.py"

app = modal.App("miles-agent-server-deepeyes-v1")
# W&B key for online logging. Set AGENT_SERVER_WANDB_SECRET="" to run without one (metrics stay in the logs).
WANDB_SECRET = os.environ.get("AGENT_SERVER_WANDB_SECRET", "miles-wandb")
volume = modal.Volume.from_name("miles-agent-proxy-swe", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.from_registry(MILES_IMAGE)
    .entrypoint([])
    .add_local_dir(
        LOCAL_MILES / "agent_server/sglang_patch", "/root/sglang_patch", copy=True, ignore=["**/__pycache__"]
    )
    .run_commands("sh /root/sglang_patch/apply.sh /sgl-workspace/sglang")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf"})
    # Local Miles checkout, synced over the image's copy at container start.
    .add_local_dir(
        LOCAL_MILES,
        "/root/miles_src",
        ignore=[".git", "**/__pycache__", "**/*.pyc", "miles.egg-info", "tests/snapshots", "docs", "charts"],
    )
)


def sync_local_miles() -> str:
    target = "/root/miles"
    subprocess.run(["rsync", "-a", "/root/miles_src/", f"{target}/"], check=True)
    return target


@app.function(image=image, cpu=8.0, memory=65536, timeout=4 * 3600, volumes={VOL: volume})
def prepare():
    """Both checkpoints, the v1 split under data/deepeyes_v1 and V* Bench under data/vstar_bench."""
    from huggingface_hub import snapshot_download

    for model in (POLICY, JUDGE):
        snapshot_download(f"Qwen/{model}", local_dir=f"{VOL}/models/{model}")
    miles = sync_local_miles()
    env = {**os.environ, "PYTHONPATH": miles}
    for script, out, done, extra in (
        ("deepeyes_v1/prepare_data.py", "deepeyes_v1", "train.jsonl", ["--processor", f"{VOL}/models/{POLICY}"]),
        ("deepeyes/prepare_vstar.py", "vstar_bench", "test.jsonl", []),
    ):
        if not Path(f"{VOL}/data/{out}/{done}").exists():
            subprocess.run(
                ["python", f"examples/experimental/{script}", "--out", f"{VOL}/data/{out}", *extra],
                cwd=miles,
                env=env,
                check=True,
            )
    volume.commit()


@app.function(image=image, cpu=8.0, memory=32768, timeout=3600, volumes={VOL: volume})
def check():
    miles = sync_local_miles()
    env = {**os.environ, "PYTHONPATH": miles, "AGENT_SERVER_TEST_PROCESSOR": f"{VOL}/models/{POLICY}"}
    # The SGLang that Python imports carries this checkout's patch: the same continuous token code, and
    # the hooks that call it.
    srt = "/sgl-workspace/sglang/python/sglang/srt"
    subprocess.run(
        f"python -c 'import importlib.util; print(importlib.util.find_spec(\"sglang\").origin)' && "
        f"diff -r -x __pycache__ agent_server/sglang_patch/continuous_token {srt}/continuous_token && "
        f"grep -c '_continue_prompt\\|_continuation\\|prompt_edit' {srt}/managers/tokenizer_manager.py "
        f"{srt}/entrypoints/openai/serving_chat.py && echo 'sglang patch: applied, continuous_token identical'",
        shell=True,
        cwd=miles,
        check=True,
    )
    for cmd in (
        f"python agent_server/experiments/deepeyes_v1/check_tokens.py --model {VOL}/models/{POLICY} "
        f"--data {VOL}/data/deepeyes_v1/val.jsonl {VOL}/data/vstar_bench/test.jsonl",
        "python -c 'import miles_plugins.megatron_bridge'",
        "python -m pytest --confcutdir=agent_server agent_server/tests -q",
    ):
        print(f"$ {cmd}", flush=True)
        subprocess.run(shlex.split(cmd), cwd=miles, env=env, check=True)


@app.function(
    image=image,
    gpu=["H200:8", "H100:8"],
    cpu=32.0,
    memory=262144,
    timeout=2 * 3600,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)] if WANDB_SECRET else [],
)
def smoke():
    """Three steps of 8 prompts, evaluated before the first and after the second on 16 rows per eval set.

    The rollouts are dumped, and the first trained episode is printed to check the prompt the recipe's
    template built (the tool schema in the system turn, crops inside <tool_response>).
    """
    import torch
    from transformers import AutoTokenizer

    run_name = "agent-server-deepeyes-v1-smoke"
    for name, source in (("deepeyes_v1", "val.jsonl"), ("vstar_bench", "test.jsonl")):
        subprocess.run(
            f"head -n 16 {VOL}/data/{name}/{source} > {VOL}/data/{name}/smoke.jsonl", shell=True, check=True
        )
    dump = f"{VOL}/runs/{run_name}/rollouts/{{rollout_id}}.pt"
    _run(
        f"--run-name {run_name} --num-rollout 3 --rollout-batch-size 8 --eval-interval 2 --save-interval 3 "
        "--val-prompt-data data/deepeyes_v1/smoke.jsonl --vstar-prompt-data data/vstar_bench/smoke.jsonl "
        f"--extra-args {shlex.quote(f'--save-debug-rollout-data {dump}')}"
    )
    samples = torch.load(dump.format(rollout_id=0), weights_only=False)["samples"]
    sample = max(samples, key=lambda s: s["tokens"].count(151657))  # the one with the most <tool_call>s
    text = AutoTokenizer.from_pretrained(f"{VOL}/models/{POLICY}").decode(sample["tokens"])
    print(re.sub(r"(<\|image_pad\|>)+", lambda m: f"<|image_pad|>x{len(m.group(0)) // 13}", text), flush=True)
    assert "# Tools" in text and "image_zoom_in_tool" in text.split("<|im_end|>")[0]


@app.function(
    image=image,
    gpu=["H200:8", "H100:8"],
    cpu=32.0,
    memory=262144,
    timeout=12 * 3600,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)] if WANDB_SECRET else [],
)
def train(launcher_args: str = ""):
    _run(launcher_args)


def _run(launcher_args: str) -> None:
    miles = sync_local_miles()
    # Modal's gVisor hands out ephemeral ports from 16000, over the 20000+ block Miles allocates server
    # ports from: an outgoing connection took an engine's port between its allocation and its bind, and the
    # run waited for that engine forever. Use Linux's default range, which Miles assumes.
    Path("/proc/sys/net/ipv4/ip_local_port_range").write_text("32768 60999")

    def commit_periodically():
        while True:
            time.sleep(600)
            try:
                volume.commit()
            except Exception as exc:
                print(f"[modal_app] volume commit failed: {exc}", flush=True)

    threading.Thread(target=commit_periodically, daemon=True).start()
    launcher = subprocess.Popen(["python", LAUNCHER, *shlex.split(launcher_args)], cwd=miles)
    judge = None
    try:
        judge = _start_judge_after_cleanup(launcher)
        if launcher.wait() != 0:
            raise RuntimeError(f"launcher exited with {launcher.returncode}")
    finally:
        for process in (launcher, judge):
            if process is not None and process.poll() is None:
                process.terminate()
        volume.commit()


def _start_judge_after_cleanup(launcher: subprocess.Popen) -> subprocess.Popen:
    """The launcher first kills every SGLang process on the node, then starts Ray: start the judge after that.

    Ray gets 7 GPUs (``--num-gpus-per-node 7``), so the judge takes the eighth.
    """
    while subprocess.run(["ray", "status"], capture_output=True).returncode != 0:
        if launcher.poll() is not None:
            raise RuntimeError(f"launcher exited with {launcher.returncode} before Ray started")
        time.sleep(5)
    command = (
        f"python -m sglang.launch_server --model-path {VOL}/models/{JUDGE} --served-model-name judge "
        f"--port {JUDGE_PORT} --mem-fraction-static 0.8 --context-length 4096 --log-level warning"
    )
    return subprocess.Popen(shlex.split(command), env={**os.environ, "CUDA_VISIBLE_DEVICES": "7"})
