"""Index downloaded ALFWorld games as Miles JSONL, or check one game on CPU."""

import argparse
import json
from pathlib import Path


def prepare_data(data_dir: Path, output: Path, split: str, limit: int | None = None) -> int:
    root = data_dir.expanduser().resolve()
    split_dir = root / "json_2.1.1" / split
    if not split_dir.is_dir():
        split_dir = root / split
    # Follow ALFWorld's own game selection: exclude unsupported sliced/movable
    # tasks and games the dataset marks unsolvable.
    games = [
        game
        for game in sorted(split_dir.rglob("game.tw-pddl"))
        if "movable" not in game.parent.parent.name
        and "Sliced" not in game.parent.parent.name
        and json.loads(game.read_text()).get("solvable", False)
    ]
    if not games:
        raise FileNotFoundError(f"No ALFWorld games marked solvable in {split_dir}. Check the download and split.")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        games = games[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for game in games:
            annotation = game.with_name("traj_data.json")
            task = game.parent.parent.name
            if annotation.exists():
                data = json.loads(annotation.read_text())
                annotations = data.get("turk_annotations", {}).get("anns", [])
                if annotations:
                    task = annotations[0].get("task_desc", task)
            row = {
                "prompt": task,
                "metadata": {
                    "game_file": str(game),
                    "instance_id": game.parent.relative_to(split_dir).as_posix(),
                },
            }
            stream.write(json.dumps(row) + "\n")
    return len(games)


def check_env(prompt_data: Path) -> None:
    # Keep data indexing usable without installing TextWorld on the local machine.
    from examples.experimental.alfworld.agent import make_env

    with prompt_data.open() as stream:
        row = json.loads(next(stream))
    env = make_env(row["metadata"]["game_file"], max_steps=30)
    try:
        observation, infos = env.reset()
        print(observation)
        command = next(command for command in infos["admissible_commands"] if command != "help")
        observation, _, done, infos = env.step(command)
        print(f"\nCommand: {command}\n{observation}\ndone={done}, won={infos['won']}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.cache/alfworld"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid_seen", "valid_unseen"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-env", action="store_true", help="Reset and step the first indexed game on CPU.")
    args = parser.parse_args()
    count = prepare_data(args.data_dir, args.output, args.split, args.limit)
    print(f"Wrote {count} games to {args.output}")
    if args.check_env:
        check_env(args.output)


if __name__ == "__main__":
    main()
