"""Build the V* Bench eval set (craigwu/vstar_bench: 191 multiple-choice questions on high-resolution images).

Writes ``<out>/test.jsonl`` with the same row layout as the training set. Images keep their original
resolution: the harness shows the model a view of at most 1M pixels but crops zoom-ins from the
original, which is where V* Bench's small targets are. The prompt is the question and its options
followed by the DeepEyes instruction. ``metadata.reward = "accuracy"`` makes the harness report
plain accuracy (the chosen letter equals the label) as the episode's reward.

Example:
  python examples/experimental/deepeyes/prepare_vstar.py --out /root/datasets/vstar_bench
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "craigwu/vstar_bench"
# The instruction the DeepEyes training prompts end with.
DEEPEYES_INSTRUCTION = (
    "Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  "
    "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            allow_patterns=["test_questions.jsonl", "direct_attributes/*", "relative_position/*"],
        )
    )
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(source / "test_questions.jsonl")]
    with open(out / "test.jsonl", "w") as f:
        for row in rows:
            image_path = out / "images" / row["image"].replace("/", "__")
            shutil.copyfile(source / row["image"], image_path)
            lines = row["text"].split("\n")
            question = "\n".join(line for line in lines if not line.startswith("Answer with the option"))
            options = {m.group(1): m.group(2).strip() for m in map(re.compile(r"\(([A-Z])\) (.*)").match, lines) if m}
            prompt = f"{question}\nAnswer with the option's letter from the given choices.\n{DEEPEYES_INSTRUCTION}"
            metadata = {
                "image": str(image_path),
                "question": question,
                "answer": row["label"],
                "options": options,
                "category": row["category"],
                "reward": "accuracy",
            }
            f.write(json.dumps({"prompt": prompt, "metadata": metadata}) + "\n")
    print(f"wrote {len(rows)} V* Bench questions to {out}")


if __name__ == "__main__":
    main()
