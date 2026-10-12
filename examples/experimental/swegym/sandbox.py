"""E2B templates and disposable SWE-Gym workspaces. No trainer imports required."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from data_utils import REWARD_MARKER

CLAUDE_VERSION = "2.1.207"
CLAUDE_SHA256 = "85e7e988a392d859f90802ca21fb26e89d3c9ab527f5ed0b08df3955e34d5c83"
CPU_COUNT = 2
MEMORY_MB = 4096


def api_key() -> str:
    value = os.environ.get("E2B_API_KEY", "").strip()
    if value:
        return value
    return Path(os.environ.get("E2B_API_KEY_FILE", "~/.config/e2b/api_key")).expanduser().read_text().strip()


def build_commands(task: dict) -> list[str]:
    return [
        "apt-get update && apt-get install -y curl ca-certificates git patch ripgrep",
        f"curl -fSL --retry 3 https://downloads.claude.ai/claude-code-releases/{CLAUDE_VERSION}/linux-x64/claude -o /usr/local/bin/claude && echo '{CLAUDE_SHA256}  /usr/local/bin/claude' | sha256sum -c - && chmod 755 /usr/local/bin/claude && claude --version",
        task["sanitize"],
        "id miles >/dev/null 2>&1 || useradd -m -s /bin/bash miles",
        "chown -R miles:miles /testbed && mkdir -p /home/miles/.claude && printf '%s' '{\"hasCompletedOnboarding\":true}' > /home/miles/.claude.json && chown -R miles:miles /home/miles",
    ]


def template_alias(task: dict) -> str:
    inputs = [task["image"], build_commands(task), CPU_COUNT, MEMORY_MB]
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    return f"miles-swegym-{task['instance_id'].lower().replace('_', '-')}-{digest}"


def build_template(task: dict) -> str:
    from e2b import Template, default_build_logger

    alias = template_alias(task)
    key = api_key()
    if not Template.alias_exists(alias, api_key=key):
        template = Template().from_image(task["image"]).set_user("root")
        for command in build_commands(task):
            template = template.run_cmd(command)
        Template.build(
            template,
            alias,
            cpu_count=CPU_COUNT,
            memory_mb=MEMORY_MB,
            api_key=key,
            on_build_logs=default_build_logger(),
        )
    return alias


async def create_sandbox(alias: str, *, task_id: str, role: str, network: dict | None = None):
    from e2b import AsyncSandbox
    from e2b.exceptions import RateLimitException

    for attempt in range(5):
        try:
            return await AsyncSandbox.create(
                template=alias,
                timeout=3000,
                api_key=api_key(),
                metadata={"owner": "miles-swegym", "task": task_id, "role": role, "experiment": os.getenv("SWEGYM_EXPERIMENT_ID", "preflight")},
                network=network,
            )
        except RateLimitException:
            if attempt == 4:
                raise
            await asyncio.sleep(min(2**attempt, 16))


def parse_verdict(stdout: str, *, allow_collection_failure: bool = False) -> dict:
    lines = [line[len(REWARD_MARKER) :] for line in stdout.splitlines() if line.startswith(REWARD_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Evaluator did not return exactly one official verdict")
    result = json.loads(lines[0])
    tail = result.get("test_log_tail", "")
    collection_failure = (
        allow_collection_failure
        and result.get("resolved") is False
        and result.get("failure_reason") == "test_log_parse_failure"
        and "test session starts" in tail
        and "ERROR collecting " in tail
        and "error during collection" in tail
    )
    if (result["parsed_tests"] == 0 and not collection_failure) or result["fail_to_pass_total"] == 0:
        raise RuntimeError(f"Invalid evaluator result: {result}")
    return result


async def evaluate(task: dict, alias: str, patch: str) -> dict:
    """Agent sees neither the gold tests nor this freshly created sandbox."""
    from e2b.sandbox.commands.command_handle import CommandExitException

    start = time.monotonic()
    sb = await create_sandbox(alias, task_id=task["instance_id"], role="verifier")
    try:
        if patch.strip():
            await sb.files.write("/tmp/candidate.patch", patch, user="root")
            try:
                await sb.commands.run("cd /testbed && git apply --binary /tmp/candidate.patch", user="root", timeout=30)
            except CommandExitException:
                return {
                    "resolved": False,
                    "failure_reason": "patch_apply_failed",
                    "eval_time": time.monotonic() - start,
                }
        result = await sb.commands.run(task["evaluate"], user="root", timeout=360)
        verdict = parse_verdict(result.stdout, allow_collection_failure=bool(patch.strip()))
        if verdict["parsed_tests"] == 0:
            # A source patch can prevent pytest collection. Verify the same task
            # in a fresh unpatched sandbox before treating that as reward zero;
            # an unavailable runner/parser must still fail, never become reward.
            control = await evaluate(task, alias, "")
            verdict = {
                **verdict,
                "failure_reason": "candidate_test_collection_failure",
                "unpatched_control_parsed_tests": control["parsed_tests"],
            }
        return {**verdict, "eval_time": time.monotonic() - start}
    finally:
        await sb.kill()
