"""Modal app for the Claude Code + SWE-Gym / SWE-bench Verified RL experiment on agent_server.

Everything runs in the user's Modal workspace:
- one GPU container (8 GPUs) runs Miles disaggregated: 4 GPUs train, 4 GPUs serve rollouts;
- the agent_server replicas inside it are exposed through Modal tunnels;
- every Claude Code trial runs in its own Modal sandbox, created by Harbor from the task image.

Usage (from the miles repo root):
    modal run agent_server/experiments/swe_claude_code/modal_app.py::inspect_image
    modal run agent_server/experiments/swe_claude_code/modal_app.py::download_model
"""

import os
import subprocess
from pathlib import Path

import modal

MILES_IMAGE = "radixark/miles:dev-202610102218"
HARBOR_REF = "git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
# Only meaningful on the machine that launches the app; inside the container this file is /root/modal_app.py.
LOCAL_MILES = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/miles_src")

app = modal.App("miles-agent-server-swe")
# Created before the rename to agent server; it holds the models, data and runs.
volume = modal.Volume.from_name("miles-agent-proxy-swe", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.from_registry(MILES_IMAGE)
    .entrypoint([])
    # agent_server needs its SGLang patch: continuous token runs inside SGLang.
    .add_local_dir(
        LOCAL_MILES / "agent_server/sglang_patch", "/root/sglang_patch", copy=True, ignore=["**/__pycache__"]
    )
    .run_commands(
        "sh /root/sglang_patch/apply.sh /sgl-workspace/sglang",
        # The image's Python is a venv under /opt/sglang; install into it, not the system interpreter.
        "python -m pip install -q uv",
        f"python -m uv pip install --python \"$(command -v python)\" 'harbor[modal] @ {HARBOR_REF}' 'modal>=1.5.5'",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf"})
    # Local Miles checkout (agent_server + hooks); synced over the image's copy at container start.
    .add_local_dir(
        LOCAL_MILES,
        "/root/miles_src",
        ignore=[".git", "**/__pycache__", "**/*.pyc", "miles.egg-info", "tests/snapshots", "docs", "charts"],
    )
)


def sync_local_miles() -> str:
    """Copy the mounted local checkout over the image's Miles install and return its path."""
    target = "/root/miles"
    subprocess.run(["rsync", "-a", "/root/miles_src/", f"{target}/"], check=True)
    return target


@app.function(image=image, timeout=3600, volumes={VOL: volume})
def inspect_image():
    for cmd in (
        "ls /root; which python; pip show miles sglang megatron-core harbor modal transformers pydantic httpx fastapi 2>/dev/null | grep -E '^(Name|Version|Location|Editable)'",
        'python -c \'import sglang.srt.entrypoints.anthropic.serving as s; print("anthropic converters:", hasattr(s, "convert_to_chat_completion_request"))\'',
        "ls /root/Megatron-LM 2>/dev/null | head -3; nvcc --version 2>/dev/null | tail -1; python -c 'import torch; print(torch.__version__)'",
        "df -h /tmp /root | tail -2; nproc; free -g | head -2",
    ):
        print(f"$ {cmd}\n{subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout}")


@app.function(image=image, timeout=3600, volumes={VOL: volume})
def download_model(repo_id: str = "Qwen/Qwen3.5-4B"):
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, local_dir=f"{VOL}/models/{repo_id.split('/')[-1]}")
    volume.commit()
    print("downloaded to", path, os.listdir(path))


@app.function(
    image=image,
    # H200 first for memory headroom on 64K-token trajectories; H100 when no H200 node is free.
    gpu=["H200:8", "H100:8"],
    cpu=32.0,
    memory=196608,
    timeout=24 * 3600,
    volumes={VOL: volume},
    # Modal token: Harbor creates the task sandboxes in this workspace. W&B key: online logging.
    secrets=[modal.Secret.from_name("miles-modal-sandbox-token"), modal.Secret.from_name("miles-wandb")],
)
def train(launcher_args: str = "", agent_server_workers: int = 4, agent_server_port: int = 30000):
    """Run run_swe_claude_code.py with each agent server replica reachable from the sandboxes through a tunnel."""
    import contextlib
    import json
    import shlex
    import threading
    import time

    miles = sync_local_miles()

    def commit_periodically():
        # Checkpoints and dumps land on the volume; commit them while the run is going, not only at exit.
        while True:
            time.sleep(600)
            try:
                volume.commit()
            except Exception as exc:
                print(f"[modal_app] volume commit failed: {exc}", flush=True)

    threading.Thread(target=commit_periodically, daemon=True).start()

    ports = [agent_server_port + i for i in range(agent_server_workers)]
    with contextlib.ExitStack() as stack:
        public_urls = {str(port): stack.enter_context(modal.forward(port)).url for port in ports}
        print(f"[modal_app] agent server replica tunnels: {public_urls}", flush=True)
        cmd = [
            "python",
            f"{miles}/agent_server/experiments/swe_claude_code/run_swe_claude_code.py",
            "--agent-server-workers",
            str(agent_server_workers),
            "--agent-server-port",
            str(agent_server_port),
            *shlex.split(launcher_args),
        ]
        env = {**os.environ, "AGENT_SERVER_PUBLIC_URLS": json.dumps(public_urls)}
        try:
            subprocess.run(cmd, cwd=miles, env=env, check=True)
        finally:
            volume.commit()
