"""Build the τ²-bench retail and airline data.

- Training: the RL split of inclusionAI/AReaL-tau2-data, retail and airline (563 + 1,148 tasks). The
  tasks are synthetic, each with a user scenario, the expected actions and its own database variant.
- Evaluation: the official τ²-bench splits of the same domains, on the domains' own databases: test
  (retail 40, airline 20) and train (retail 74, airline 30). This recipe does not train on the official
  train split (104 tasks are too few to train on without repeating them), so it is a second held-out
  set that makes the evaluation less noisy.

Writes ``<out>/train.jsonl``, ``<out>/test.jsonl``, ``<out>/official_train.jsonl`` and the database
variants under ``<out>/db``. Each row's ``prompt`` is the task id; ``metadata`` holds the domain, the
task as tau2 ``Task`` JSON and the database file (``None`` for the domain's own). Every task is
validated against tau2's ``Task`` model. The training rows are shuffled with ``--seed``.

Example:
  TAU2_DATA_DIR=/root/tau2-bench/data python examples/experimental/tau2/prepare_data.py --out /root/datasets/tau2
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from tau2.data_model.tasks import Task
from tau2.registry import registry

REPO = "inclusionAI/AReaL-tau2-data"
DOMAINS = ("retail", "airline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "db").mkdir(parents=True, exist_ok=True)
    train = [_train_row(json.loads(line), out) for line in open(_download("tau2_rl_train.jsonl"))]
    train = [row for row in train if row is not None]
    random.Random(args.seed).shuffle(train)
    official = {
        split: [
            _row(domain, task.model_dump(mode="json"), db=None)
            for domain in DOMAINS
            for task in registry.get_tasks_loader(domain)(split)
        ]
        for split in ("test", "train")
    }
    for name, rows in (("train", train), ("test", official["test"]), ("official_train", official["train"])):
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        counts = {domain: sum(row["metadata"]["domain"] == domain for row in rows) for domain in DOMAINS}
        print(f"{name}: {len(rows)} tasks {counts}")


def _train_row(raw: dict, out: Path) -> dict | None:
    domain = raw["user_scenario"]["instructions"]["domain"]
    if domain not in DOMAINS:
        return None
    db = out / "db" / Path(raw["db_path"]).name
    if not db.exists():
        shutil.copyfile(_download(raw["db_path"]), db)
    task = {
        "id": raw["id"],
        "description": raw["description"],
        "user_scenario": raw["user_scenario"],
        "evaluation_criteria": json.loads(raw["evaluation_criteria"]),
    }
    return _row(domain, task, db=str(db))


def _row(domain: str, task: dict, *, db: str | None) -> dict:
    Task.model_validate(task)
    return {"prompt": task["id"], "metadata": {"domain": domain, "task": task, "db": db}}


def _download(path: str) -> str:
    return hf_hub_download(REPO, path, repo_type="dataset")


if __name__ == "__main__":
    main()
