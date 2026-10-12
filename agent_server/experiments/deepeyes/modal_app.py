"""Modal app for the DeepEyes run on agent_server (examples/experimental/deepeyes).

The image is the Miles image with agent_server/sglang_patch applied to its SGLang (needed by every
agent_server run: continuous token runs inside SGLang).

- prepare: download Qwen3-VL-4B-Instruct, 4000 DeepEyes training rows and V* Bench to the volume (CPU);
- tests: agent_server's tests and the Miles tests next to the changed code (CPU);
- probe_bbox: which coordinate frame the policy's zoom-in boxes use (one GPU);
- smoke: a patched SGLang on one GPU, checked turn by turn against the one-shot HF processor
  (agent_server/tests/test_multimodal.py with AGENT_SERVER_TEST_SGLANG_URL);
- train: the run, 2 GPUs training and 2 GPUs serving rollouts.

Usage (from the miles repo root, with a Modal CLI >= 1.6):
    modal run agent_server/experiments/deepeyes/modal_app.py::prepare
    modal run agent_server/experiments/deepeyes/modal_app.py::smoke
    modal run --detach agent_server/experiments/deepeyes/modal_app.py::train --launcher-args "--num-rollout 30"
"""

import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

MILES_IMAGE = "radixark/miles:dev-202610102218"
MODEL = "Qwen3-VL-4B-Instruct"
# Only meaningful on the machine that launches the app; inside the container this file is /root/modal_app.py.
LOCAL_MILES = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/miles_src")

app = modal.App("miles-agent-server-deepeyes")
# W&B key for online logging. Set AGENT_SERVER_WANDB_SECRET="" to run without one (metrics stay in the logs).
WANDB_SECRET = os.environ.get("AGENT_SERVER_WANDB_SECRET", "miles-wandb")
# Created before the rename to agent server; it holds the models, data and runs.
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


@app.function(image=image, cpu=8.0, memory=32768, timeout=3 * 3600, volumes={VOL: volume})
def prepare():
    """The model, 4000 DeepEyes training rows under data/deepeyes and V* Bench under data/vstar_bench."""
    from huggingface_hub import snapshot_download

    snapshot_download(f"Qwen/{MODEL}", local_dir=f"{VOL}/models/{MODEL}")
    miles = sync_local_miles()
    env = {**os.environ, "PYTHONPATH": miles}
    for script, out, done in (
        ("prepare_data.py", "deepeyes", "train.jsonl"),
        ("prepare_vstar.py", "vstar_bench", "test.jsonl"),
    ):
        if not Path(f"{VOL}/data/{out}/{done}").exists():
            subprocess.run(
                ["python", f"examples/experimental/deepeyes/{script}", "--out", f"{VOL}/data/{out}"],
                cwd=miles,
                env=env,
                check=True,
            )
    volume.commit()


@app.function(image=image, cpu=8.0, memory=32768, timeout=3600, volumes={VOL: volume})
def tests():
    miles = sync_local_miles()
    env = {
        **os.environ,
        "PYTHONPATH": miles,
        "AGENT_SERVER_TEST_TOKENIZER": f"{VOL}/models/{MODEL}",
        "AGENT_SERVER_TEST_PROCESSOR": f"{VOL}/models/{MODEL}",
    }
    for cmd in (
        # The Miles hooks import agent_server lazily; nothing else imports them without a cluster.
        "python -c 'import agent_server.miles_adapter, miles.ray.specs.entrypoint, miles.ray.specs.rollout, "
        'miles.ray.placement_group, miles.ray.rollout.rollout_executor; print("hooks import ok")\'',
        "python -m pytest --confcutdir=agent_server agent_server/tests -q",
        "python -m pytest -q tests/fast/rollout/session/test_samples_codec.py tests/fast/router/test_session_samples_op_v2.py "
        "tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py tests/fast/utils/test_processing_utils.py "
        "tests/fast/backends/training_utils/test_rollout_mask_sums.py tests/fast/backends/training_utils/test_input_loss_masks.py",
    ):
        print(f"$ {cmd}", flush=True)
        subprocess.run(shlex.split(cmd), cwd=miles, env=env, check=False)


@app.function(image=image, gpu="H100", timeout=3600, volumes={VOL: volume})
def smoke():
    """The multimodal equivalence test, with every prompt expanded by a real patched SGLang."""
    miles = sync_local_miles()
    model = f"{VOL}/models/{MODEL}"
    server = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--port",
            "30000",
            "--mem-fraction-static",
            "0.6",
        ]
    )
    try:
        _wait_healthy("http://127.0.0.1:30000/health", timeout=900)
        env = {
            **os.environ,
            "PYTHONPATH": miles,
            "AGENT_SERVER_TEST_SGLANG_URL": "http://127.0.0.1:30000",
            "AGENT_SERVER_TEST_PROCESSOR": model,
        }
        subprocess.run(
            ["python", "-m", "pytest", "--confcutdir=agent_server", "agent_server/tests/test_multimodal.py", "-v"],
            cwd=miles,
            env=env,
            check=True,
        )
    finally:
        server.terminate()


@app.function(image=image, gpu="H100", timeout=3600, volumes={VOL: volume})
def probe_bbox():
    """The policy's first-turn zoom-in boxes on training and V* Bench images (probe_bbox.py)."""
    miles = sync_local_miles()
    model = f"{VOL}/models/{MODEL}"
    server = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--port",
            "30000",
            "--mem-fraction-static",
            "0.7",
        ]
        + ["--tool-call-parser", "qwen25"]
    )
    try:
        _wait_healthy("http://127.0.0.1:30000/health", timeout=900)
        os.makedirs(f"{VOL}/probe", exist_ok=True)
        subprocess.run(
            ["python", "examples/experimental/deepeyes/probe_bbox.py", "--url", "http://127.0.0.1:30000"]
            + ["--data", f"{VOL}/data/deepeyes/train.jsonl", f"{VOL}/data/vstar_bench/test.jsonl"]
            + ["--out", f"{VOL}/probe/bbox.json"],
            cwd=miles,
            env={**os.environ, "PYTHONPATH": miles},
            check=True,
        )
        volume.commit()
    finally:
        server.terminate()


@app.function(
    image=image,
    gpu=["H100:4", "H200:4"],
    cpu=16.0,
    memory=131072,
    timeout=12 * 3600,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)] if WANDB_SECRET else [],
)
def train(launcher_args: str = ""):
    import threading

    miles = sync_local_miles()

    def commit_periodically():
        while True:
            time.sleep(600)
            try:
                volume.commit()
            except Exception as exc:
                print(f"[modal_app] volume commit failed: {exc}", flush=True)

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        subprocess.run(
            ["python", "examples/experimental/deepeyes/run_qwen3_vl_4b_deepeyes.py", *shlex.split(launcher_args)],
            cwd=miles,
            check=True,
        )
    finally:
        volume.commit()


def _wait_healthy(url: str, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError(f"{url} did not become healthy in {timeout}s")
