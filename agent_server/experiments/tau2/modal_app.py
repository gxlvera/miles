"""Modal app for the τ²-bench run (examples/experimental/tau2: retail + airline, Qwen3-4B-Instruct-2507).

The image is the Miles image with agent_server/sglang_patch applied to its SGLang and the tau2 fork
AReaL trains with installed. One 8-GPU node: 2 GPUs train, 4 serve the policy, the last 2 serve the
user simulator (Qwen3-30B-A3B-Instruct-2507, TP 2).

- prepare: the policy and simulator checkpoints, and the τ² data under data/tau2 (CPU);
- check: the patched SGLang, tau2, the data and the agent server tests (CPU);
- smoke: a 3-step run on a small eval set, with an evaluation and a checkpoint save;
- train: the run.

Usage (from the miles repo root, with a Modal CLI >= 1.6):
    modal run agent_server/experiments/tau2/modal_app.py::prepare
    modal run agent_server/experiments/tau2/modal_app.py::check
    modal run agent_server/experiments/tau2/modal_app.py::smoke
    modal run --detach agent_server/experiments/tau2/modal_app.py::train --launcher-args "--num-rollout 30"
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
POLICY = "Qwen3-4B-Instruct-2507"
USER_SIMULATOR = "Qwen3-30B-A3B-Instruct-2507"
USER_PORT = 18902
# dhh1995/tau2-bench, branch dhh/async-and-custom-completion: tau2-bench with async agents and pluggable
# completion functions, as AReaL's tau2 example uses it.
TAU2_COMMIT = "e33e951"
TAU2_DIR = "/root/tau2-bench"
# Only meaningful on the machine that launches the app; inside the container this file is /root/modal_app.py.
LOCAL_MILES = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/miles_src")
LAUNCHER = "examples/experimental/tau2/run_qwen3_4b_tau2.py"

app = modal.App("miles-agent-server-tau2")
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
    # tau2 without its dependencies, then only the ones it imports that the image lacks, so nothing the
    # image already has (pydantic, fastapi, httpx, openai, ...) changes version.
    .run_commands(
        f"git clone -q -b dhh/async-and-custom-completion https://github.com/dhh1995/tau2-bench.git {TAU2_DIR}"
        f" && git -C {TAU2_DIR} checkout -q {TAU2_COMMIT}",
        f"python -m pip install -q --no-deps -e {TAU2_DIR}",
        "python -m pip install -q --no-deps addict 'deepdiff>=8.4' orderly-set cachebox docstring-parser"
        " python-dotenv loguru toml 'litellm>=1.65,<1.78'",
        # litellm's own dependencies, by name only: pip skips the installed ones.
        "python -m pip install -q --no-deps aiohttp click fastuuid httpx importlib-metadata jinja2 jsonschema"
        " openai tiktoken tokenizers",
        # Fails the build, not a run, when a dependency is missing.
        f"TAU2_DATA_DIR={TAU2_DIR}/data python -c 'import litellm, tau2.agent.llm_agent, tau2.evaluator.evaluator,"
        " tau2.orchestrator.orchestrator, tau2.registry, tau2.user.user_simulator'",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf", "TAU2_DATA_DIR": f"{TAU2_DIR}/data"})
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


@app.function(image=image, cpu=8.0, memory=32768, timeout=3 * 3600, volumes={VOL: volume})
def prepare():
    """Both checkpoints and the τ² data under data/tau2."""
    from huggingface_hub import snapshot_download

    for model in (POLICY, USER_SIMULATOR):
        snapshot_download(f"Qwen/{model}", local_dir=f"{VOL}/models/{model}")
    miles = sync_local_miles()
    if not Path(f"{VOL}/data/tau2/official_train.jsonl").exists():
        subprocess.run(
            ["python", "examples/experimental/tau2/prepare_data.py", "--out", f"{VOL}/data/tau2"],
            cwd=miles,
            env={**os.environ, "PYTHONPATH": miles},
            check=True,
        )
    volume.commit()


@app.function(image=image, cpu=8.0, memory=32768, timeout=3600, volumes={VOL: volume})
def check(limit: int = 0):
    """The patched SGLang, tau2 and its data (``limit``: only that many tasks per split), and the agent server
    tests (CPU)."""
    miles = sync_local_miles()
    env = {
        **os.environ,
        "PYTHONPATH": miles,
        "AGENT_SERVER_TEST_INSTRUCT_TOKENIZER": f"{VOL}/models/{POLICY}",
    }
    srt = "/sgl-workspace/sglang/python/sglang/srt"
    subprocess.run(
        f"diff -r -x __pycache__ agent_server/sglang_patch/continuous_token {srt}/continuous_token && "
        f"grep -c '_continue_prompt\\|_continuation\\|prompt_edit' {srt}/managers/tokenizer_manager.py "
        f"{srt}/entrypoints/openai/serving_chat.py && echo 'sglang patch: applied, continuous_token identical'",
        shell=True,
        cwd=miles,
        check=True,
    )
    for cmd in (
        f"python agent_server/experiments/tau2/check_data.py --data {VOL}/data/tau2"
        + (f" --limit {limit}" if limit else ""),
        "python -m pytest --confcutdir=agent_server agent_server/tests -q",
    ):
        print(f"$ {cmd}", flush=True)
        subprocess.run(shlex.split(cmd), cwd=miles, env=env, check=True)


@app.function(
    image=image,
    gpu=["H100:8", "H200:8"],
    cpu=32.0,
    memory=262144,
    timeout=2 * 3600,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)] if WANDB_SECRET else [],
)
def smoke():
    """Three steps of 8 prompts, evaluated before the first and after the second on 4 tasks per domain.

    The rollouts are dumped, and the longest trained episode is printed to check the conversation the
    agent server built: user turns, tool calls and tool results in one sequence.
    """
    import json

    import torch
    from transformers import AutoTokenizer

    run_name = "agent-server-tau2-smoke"
    rows = [json.loads(line) for line in open(f"{VOL}/data/tau2/test.jsonl")]
    smoke_rows = [r for d in ("retail", "airline") for r in [r for r in rows if r["metadata"]["domain"] == d][:4]]
    with open(f"{VOL}/data/tau2/smoke.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in smoke_rows)
    dump = f"{VOL}/runs/{run_name}/rollouts/{{rollout_id}}.pt"
    _run(
        f"--run-name {run_name} --num-rollout 3 --rollout-batch-size 8 --eval-interval 2 --save-interval 3 "
        "--n-samples-per-eval-prompt 1 --eval-prompt-data data/tau2/smoke.jsonl --eval-official-train-data '' "
        f"--extra-args {shlex.quote(f'--save-debug-rollout-data {dump}')}"
    )
    samples = torch.load(dump.format(rollout_id=0), weights_only=False)["samples"]
    sample = max(samples, key=lambda s: len(s["tokens"]))
    text = AutoTokenizer.from_pretrained(f"{VOL}/models/{POLICY}").decode(sample["tokens"])
    print(re.sub(r"\n{3,}", "\n\n", text[-12000:]), flush=True)
    print(
        "reward",
        sample["reward"],
        "tokens",
        len(sample["tokens"]),
        "metadata",
        {k: v for k, v in sample["metadata"].items() if k in ("num_turns", "num_tool_calls", "db_match", "user_stop")},
    )
    assert "<tool_call>" in text and "<tool_response>" in text and text.count("<|im_start|>user") > 2


@app.function(
    image=image,
    gpu=["H100:8", "H200:8"],
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
    # ports from: an outgoing connection can take an engine's port between its allocation and its bind.
    # Use Linux's default range, which Miles assumes.
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
    user_simulator = None
    try:
        user_simulator = _start_user_simulator_after_cleanup(launcher)
        if launcher.wait() != 0:
            raise RuntimeError(f"launcher exited with {launcher.returncode}")
    finally:
        for process in (launcher, user_simulator):
            if process is not None and process.poll() is None:
                process.terminate()
        volume.commit()


def _start_user_simulator_after_cleanup(launcher: subprocess.Popen) -> subprocess.Popen:
    """The launcher first kills every SGLang process on the node, then starts Ray: start the simulator after that.

    Ray gets 6 GPUs (``--num-gpus-per-node 6``), so the simulator takes the last two. It runs with OpenAI
    sampling defaults, so the checkpoint's generation_config does not change tau2's temperature-0 user.
    """
    while subprocess.run(["ray", "status"], capture_output=True).returncode != 0:
        if launcher.poll() is not None:
            raise RuntimeError(f"launcher exited with {launcher.returncode} before Ray started")
        time.sleep(5)
    command = (
        f"python -m sglang.launch_server --model-path {VOL}/models/{USER_SIMULATOR} --served-model-name user "
        f"--tp 2 --port {USER_PORT} --mem-fraction-static 0.85 --context-length 32768 "
        "--sampling-defaults openai --log-level warning"
    )
    return subprocess.Popen(shlex.split(command), env={**os.environ, "CUDA_VISIBLE_DEVICES": "6,7"})
