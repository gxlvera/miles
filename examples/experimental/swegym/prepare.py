"""Prepare a small, deterministic SWE-Gym training split and host-only evaluators.

Usage: python prepare.py --input train.parquet --harness-root SWE-Bench-Package --output data
Gold patches are written only to calibration.json, never to prompts or templates.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from data_utils import (
    SWEGYM_HARNESS_COMMIT,
    build_eval_command,
    build_git_sanitizer_command,
    download_split,
    format_prompt,
    load_official_log_parser_source,
    load_parquet,
    load_test_spec_builder,
    registry_image_for_instance,
)


def prepare(source: Path, harness_root: Path, output: Path, repos: list[str], limit: int, instance_ids: list[str] | None = None) -> None:
    rows = load_parquet(source)
    selected = [row for row in rows if (not repos or row["repo"] in repos) and (not instance_ids or row["instance_id"] in instance_ids)][:limit]
    if not selected:
        raise ValueError("No tasks selected")
    make_spec = load_test_spec_builder(harness_root)
    from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS

    parser_source = load_official_log_parser_source(harness_root)
    output.mkdir(parents=True, exist_ok=True)
    prompts, tasks, calibration = [], {}, {}
    for row in selected:
        task_id = row["instance_id"]
        if not row["FAIL_TO_PASS"]:
            raise ValueError(f"{task_id}: empty FAIL_TO_PASS is not a training task")
        spec = make_spec(dict(row))
        reuse_environment = row["repo"] in {"pydantic/pydantic", "facebookresearch/hydra"}
        if reuse_environment:
            # Instance images already contain the installed editable repository.
            # These pure-Python tasks need no rebuild after a source-only patch.
            install = MAP_REPO_VERSION_TO_SPECS[row["repo"].lower()][row["version"]].get("install")
            if install:
                if spec.eval_script_list.count(install) != 1:
                    raise ValueError("Unexpected official evaluator installation structure")
                spec = replace(spec, eval_script_list=[cmd for cmd in spec.eval_script_list if cmd != install])
        task = {
            "instance_id": task_id,
            "image": registry_image_for_instance(task_id, row["repo"]),
            "base_commit": row["base_commit"],
            "repo": row["repo"],
            "sanitize": build_git_sanitizer_command(base_commit=row["base_commit"], workdir="/testbed")["cmd"],
            "evaluate": build_eval_command(
                eval_script=str(spec.eval_script),
                instance=row,
                timeout=300,
                log_parser_source=parser_source,
            ),
            "harness_commit": SWEGYM_HARNESS_COMMIT,
            "reuse_instance_environment": reuse_environment,
        }
        tasks[task_id] = task
        prompts.append({"prompt": format_prompt(row["problem_statement"]), "metadata": {"instance_id": task_id}})
        calibration[task_id] = row["patch"]
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prompts))
    (output / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(json.dumps({"tasks": len(tasks), "instances": list(tasks), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repos", nargs="*", default=["pydantic/pydantic", "facebookresearch/hydra"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--instance-ids", nargs="*")
    args = parser.parse_args()
    source = args.input or download_split("train", args.output / "source")
    prepare(source, args.harness_root, args.output, args.repos, args.limit, args.instance_ids)
