"""Prebuild templates, smoke-test, and calibrate the official evaluator.

Usage: python templates.py --data-dir data --calibrate
Both an empty patch (must fail) and the gold patch (must pass) are checked.
"""

import argparse
import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sandbox import build_template, create_sandbox, evaluate


async def verify(task: dict, alias: str, gold: str | None) -> dict:
    sb = await create_sandbox(alias, task_id=task["instance_id"], role="smoke")
    try:
        check = await sb.commands.run(
            "cd /testbed && claude --version && git status --porcelain && git rev-parse HEAD",
            user="miles",
            timeout=30,
        )
        if task["base_commit"] not in check.stdout:
            raise RuntimeError(f"Wrong base commit: {check.stdout}")
    finally:
        await sb.kill()
    result = {"smoke": True, "template": alias, "evaluator_sha256": hashlib.sha256(task["evaluate"].encode()).hexdigest()}
    if gold is not None:
        empty = await evaluate(task, alias, "")
        positive = await evaluate(task, alias, gold)
        if empty["resolved"] or not positive["resolved"]:
            raise RuntimeError(f"Calibration failed for {task['instance_id']}: empty={empty}, gold={positive}")
        result.update(empty=empty, gold=positive)
    return result


async def calibrate(tasks, aliases, gold, workers, output):
    semaphore = asyncio.Semaphore(workers)
    results = {}

    async def one(task_id, task):
        async with semaphore:
            results[task_id] = await verify(task, aliases[task_id], gold.get(task_id))
            output.write_text(json.dumps(results, indent=2))
            print(task_id, "verified", flush=True)

    async with asyncio.TaskGroup() as group:
        for task_id, task in tasks.items():
            group.create_task(one(task_id, task))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    tasks = json.loads((args.data_dir / "tasks.json").read_text())
    gold = json.loads((args.data_dir / "calibration.json").read_text()) if args.calibrate else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        aliases = dict(zip(tasks, pool.map(build_template, tasks.values()), strict=True))
    (args.data_dir / "templates.json").write_text(json.dumps(aliases, indent=2))
    asyncio.run(calibrate(tasks, aliases, gold, args.workers, args.data_dir / "preflight.json"))


if __name__ == "__main__":
    main()
