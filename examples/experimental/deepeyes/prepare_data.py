"""Build the DeepEyes training set from ChenShawn/DeepEyes-Datasets-47k.

Writes ``<out>/train.jsonl`` and the images as PNG files under ``<out>/images``. Each row's
``prompt`` is the dataset's user text without its ``<image>`` tag; ``metadata`` holds the image path,
the question and the reference answer. Images larger than ``--max-pixels`` are downscaled, which
bounds the tokens per image. Like the verl recipe, the Chart subset
(``data_v0.8_visual_toolbox_v2.parquet``) is left out.

The rows are a random sample: Miles draws training prompts without replacement within an epoch, so
a run sees no prompt twice as long as it draws fewer than ``--num-rows`` prompts.

Example:
  python examples/experimental/deepeyes/prepare_data.py --out /root/datasets/deepeyes --num-rows 4000
"""

import argparse
import io
import json
import math
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "ChenShawn/DeepEyes-Datasets-47k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", nargs="+", default=["data_0.1.2_visual_toolbox_v2.parquet"])
    parser.add_argument("--num-rows", type=int, default=4000, help="rows sampled over all files")
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for file in args.files:
        table = pq.read_table(hf_hub_download(REPO, file, repo_type="dataset"))
        rows += [(file, row) for row in table.to_pylist()]
    random.Random(args.seed).shuffle(rows)

    with open(out / "train.jsonl", "w") as f:
        for index, (file, row) in enumerate(rows[: args.num_rows]):
            image_path = out / "images" / f"{index}.png"
            _downscale(Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB"), args.max_pixels).save(
                image_path
            )
            user_text = row["prompt"][-1]["content"].replace("<image>\n", "").replace("<image>", "")
            metadata = {
                "image": str(image_path),
                "question": row["extra_info"]["question"],
                "answer": row["reward_model"]["ground_truth"],
                "data_source": row["data_source"],
                "source_file": file,
            }
            f.write(json.dumps({"prompt": user_text, "metadata": metadata}) + "\n")
    print(f"wrote {min(len(rows), args.num_rows)} of {len(rows)} rows to {out}")


def _downscale(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    main()
