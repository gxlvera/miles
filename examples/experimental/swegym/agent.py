"""Claude Code -> gxlvera agent_proxy, with fresh E2B patch evaluation."""

import asyncio
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from sandbox import create_sandbox, evaluate


def forbidden_changes(paths: list[str]) -> list[str]:
    configs = {"pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "conftest.py"}
    return [name for name in paths if any(part in {"tests", "test", ".github"} for part in Path(name).parts) or Path(name).name in configs or Path(name).name.startswith("test_")]


def public_proxy_url(base_url: str, public_url: str) -> str:
    source, target = urlsplit(base_url), urlsplit(public_url)
    if target.scheme != "https" or not target.netloc or target.path not in ("", "/"):
        raise ValueError("SWEGYM_PROXY_PUBLIC_URL must be an HTTPS origin")
    if not source.path.startswith("/sessions/"):
        raise ValueError(f"Unexpected Proxy session URL: {base_url}")
    # Modal assigns a TLS host:443 rather than preserving the worker's port.
    worker = (source.port or 30000) - 30000
    if worker < 0 or worker >= int(os.getenv("SWEGYM_PROXY_WORKERS", "1")):
        raise ValueError("Unexpected Proxy replica port")
    prefix = f"/workers/{worker}" if worker else ""
    return public_url.rstrip("/") + prefix + source.path


def claude_environment(base_url: str, token: str, max_seq_len: int = 65536) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": token,
        "ANTHROPIC_MODEL": "Qwen3.5-4B",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Qwen3.5-4B",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen3.5-4B",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "Qwen3.5-4B",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_PROMPT_CACHING": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "DISABLE_COMPACT": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "ENABLE_TOOL_SEARCH": "false",
        "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.getenv("SWEGYM_MAX_OUTPUT_TOKENS", "2048"),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(max_seq_len),
        "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS": "2000",
        "BASH_MAX_OUTPUT_LENGTH": "6000",
        "HOME": "/home/miles",
        "PATH": "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
        "CONDA_DEFAULT_ENV": "testbed",
        "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
    }


async def _run(base_url: str, prompt, request_kwargs=None, metadata=None, **kwargs) -> dict:
    from e2b.sandbox.commands.command_handle import CommandExitException

    start = time.monotonic()
    data_dir = Path(os.environ["SWEGYM_DATA_DIR"])
    task_id = metadata["instance_id"]
    task = json.loads((data_dir / "tasks.json").read_text())[task_id]
    alias = json.loads((data_dir / "templates.json").read_text())[task_id]
    endpoint = public_proxy_url(base_url, os.environ["SWEGYM_PROXY_PUBLIC_URL"])
    token = Path(os.environ["SWEGYM_GATEWAY_TOKEN_FILE"]).read_text().strip()
    trial_dir = Path(os.environ["SWEGYM_TRIALS_DIR"]) / f"{task_id}-{uuid.uuid4().hex}"
    trial_dir.mkdir(parents=True)
    # Allow only the policy endpoint: no GitHub solution retrieval or web tools.
    network = {"deny_out": ["0.0.0.0/0"], "allow_out": [urlsplit(endpoint).hostname]}
    sandbox_start = time.monotonic()
    sb = await create_sandbox(alias, task_id=task_id, role="agent", network=network)
    agent_start = time.monotonic()
    try:
        if isinstance(prompt, list):
            prompt = "\n".join(message["content"] for message in prompt)
        await sb.files.write("/tmp/problem.txt", str(prompt), user="miles")
        await sb.files.write("/tmp/miles_tool_limits.py", Path(__file__).with_name("tool_limits.py").read_text(), user="miles")
        settings = {"hooks": {"PreToolUse": [{"matcher": "Read|Grep", "hooks": [{"type": "command", "command": "python /tmp/miles_tool_limits.py"}]}]}}
        await sb.files.write("/tmp/miles_claude_settings.json", json.dumps(settings), user="miles")
        max_turns = int(os.getenv("SWEGYM_MAX_TURNS", "30"))
        budget = int(os.getenv("SWEGYM_AGENT_TIMEOUT", "2400"))
        command = (
            f"cd /testbed && timeout --signal=TERM --kill-after=10 {budget} claude --print --verbose --output-format stream-json "
            "--model Qwen3.5-4B --dangerously-skip-permissions --setting-sources user "
            "--settings /tmp/miles_claude_settings.json "
            "--tools 'Bash,Read,Edit,Write,Glob,Grep' "
            "--disallowedTools 'WebSearch,WebFetch,Agent,Task' "
            f"--max-turns {max_turns} < /tmp/problem.txt > /tmp/claude.jsonl 2> /tmp/claude.err"
        )
        agent_exit = 0
        try:
            await sb.commands.run(
                command,
                user="miles",
                envs=claude_environment(endpoint, token, int(os.getenv("SWEGYM_CONTEXT_LENGTH", "65536"))),
                timeout=budget + 30,
            )
        except CommandExitException as exc:
            agent_exit = exc.exit_code
        trace = await sb.files.read("/tmp/claude.jsonl", user="miles")
        errors = await sb.files.read("/tmp/claude.err", user="miles")
        (trial_dir / "claude.jsonl").write_text(trace)
        (trial_dir / "claude.stderr").write_text(errors)
        # Intent-to-add records new files; compare against the immutable base even
        # if the agent committed. The evaluator receives only the resulting diff.
        diff = await sb.commands.run(
            "cd /testbed && git add -N . && git diff --binary " + shlex.quote(task["base_commit"]),
            user="miles",
            timeout=30,
        )
        patch = diff.stdout
        (trial_dir / "candidate.patch").write_text(patch)
        names = await sb.commands.run(
            "cd /testbed && git diff --name-only -z " + shlex.quote(task["base_commit"]),
            user="miles",
            timeout=30,
        )
        rejected = forbidden_changes([name for name in names.stdout.split("\0") if name])
    finally:
        await sb.kill()
    agent_time = time.monotonic() - agent_start
    verdict = {"resolved": False, "failure_reason": "protected_file_modified", "paths": rejected, "eval_time": 0.0} if rejected else await evaluate(task, alias, patch)
    result = {
        "miles_session_id": urlsplit(base_url).path.split("/")[-1],
        "num_turns": len({r["message"]["id"] for line in trace.splitlines() if (r := json.loads(line)).get("type") == "assistant" and r.get("message", {}).get("id") and r["message"].get("model") != "<synthetic>"}),
        "reward": float(verdict["resolved"]),
        "exit_status": "Submitted" if agent_exit == 0 else "AgentError",
        "agent_exit_code": agent_exit,
        "eval_report": verdict,
        "trial_dir": str(trial_dir),
        "agent_metrics": {
            "total_time": time.monotonic() - start,
            "sandbox_start_time": agent_start - sandbox_start,
            "agent_run_time": agent_time,
            "eval_time": verdict["eval_time"],
        },
    }
    (trial_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


async def reward(args, samples, **kwargs):
    def value(sample):
        if "eval_report" not in sample.metadata:
            raise RuntimeError("SWE-Gym episode has no verified evaluation; refusing an infrastructure reward")
        return float(sample.metadata["reward"])

    return [value(sample) for sample in samples] if isinstance(samples, list) else value(samples)


_CONCURRENCY = asyncio.Semaphore(int(os.getenv("SWEGYM_AGENT_CONCURRENCY", "12")))


async def run(*args, **kwargs):
    async with _CONCURRENCY:
        return await _run(*args, **kwargs)
