"""Build the DeepEyes v1 train and validation sets the way seed-verl's scripts/deepeyes_v1/prepare_data.py does.

V* (``data_0.1.2_visual_toolbox_v2``) and ThinkLite (``data_thinklite_reasoning_acc``) from
ChenShawn/DeepEyes-Datasets-47k, Chart left out, images with a side under 28 px upscaled, all rows
shuffled with seed 42: the first 512 are the validation set, the rest the training set. As verl's
``filter_overlong_prompts`` does, rows whose prompt (system + user, image included) exceeds 8192
tokens are dropped from both.

Only the first ``--num-train-rows`` training rows are written; the split is shuffled, so they are a
random sample. Miles draws training prompts without replacement within an epoch, so a run sees no
prompt twice while it draws fewer.

Writes ``<out>/train.jsonl``, ``<out>/val.jsonl`` and the images under ``<out>/images``. Each row's
``prompt`` is the dataset's user message, ``<image>`` tag included; ``metadata`` holds the image
path, the question, the reference answer and the data source.

Example:
  python examples/experimental/deepeyes_v1/prepare_data.py --out /root/datasets/deepeyes_v1 \\
      --processor /root/models/Qwen2.5-VL-7B-Instruct
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from examples.experimental.deepeyes_v1.agent import SYSTEM_PROMPT, prompt_parts
from huggingface_hub import hf_hub_download
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoProcessor

REPO = "ChenShawn/DeepEyes-Datasets-47k"
SOURCES = ["data_0.1.2_visual_toolbox_v2.parquet", "data_thinklite_reasoning_acc.parquet"]
MIN_DIM = 28
VAL_SIZE = 512
SEED = 42
MAX_PROMPT_TOKENS = 8192
CHAT_TEMPLATE = Path(__file__).with_name("chat_template.jinja")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--processor", required=True, help="Qwen2.5-VL checkpoint, for the prompt-length filter")
    parser.add_argument("--num-train-rows", type=int, default=4000)
    args = parser.parse_args()

    frames = [pd.read_parquet(hf_hub_download(REPO, source, repo_type="dataset")) for source in SOURCES]
    rows = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    processor = AutoProcessor.from_pretrained(args.processor)
    processor.chat_template = CHAT_TEMPLATE.read_text()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    _write(out, "val", rows.iloc[:VAL_SIZE], processor, limit=VAL_SIZE)
    _write(out, "train", rows.iloc[VAL_SIZE:], processor, limit=args.num_train_rows)


def _write(out: Path, split: str, rows: pd.DataFrame, processor, *, limit: int) -> None:
    written = dropped = 0
    with open(out / f"{split}.jsonl", "w") as f:
        for index, row in rows.iterrows():
            if written == limit:
                break
            raw = _fix_image_bytes(row["images"][0]["bytes"])
            prompt = next(message["content"] for message in row["prompt"] if message["role"] == "user")
            if _prompt_tokens(processor, prompt, Image.open(io.BytesIO(raw))) > MAX_PROMPT_TOKENS:
                dropped += 1
                continue
            image_path = out / "images" / f"{index:05d}"
            image_path.write_bytes(raw)
            metadata = {
                "image": str(image_path),
                "question": row["extra_info"]["question"],
                "answer": row["reward_model"]["ground_truth"],
                "data_source": row["data_source"],
            }
            f.write(json.dumps({"prompt": prompt, "metadata": metadata}) + "\n")
            written += 1
    print(f"{split}: wrote {written} rows of {len(rows)}, dropped {dropped} overlong prompts")


def _fix_image_bytes(raw: bytes) -> bytes:
    """The recipe's upscaling of images with a side under 28 px (the Qwen2.5-VL processor's minimum)."""
    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    if min(width, height) >= MIN_DIM:
        return raw
    scale = (MIN_DIM + 1) / min(width, height)
    size = (max(MIN_DIM, round(width * scale)), max(MIN_DIM, round(height * scale)))
    buffer = io.BytesIO()
    image.convert("RGB").resize(size, Image.BICUBIC).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _prompt_tokens(processor, prompt: str, image: Image.Image) -> int:
    """Prompt length as verl's filter counts it: no tool schema (the agent loop adds it later)."""
    parts = prompt_parts(prompt)
    content = [{"type": "image"} if part == "<image>" else {"type": "text", "text": part} for part in parts]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    width, height = fetch_image({"image": image}).size
    num_images = parts.count("<image>")
    # One <|image_pad|> per image in the text; each expands to one token per 28x28 pixels.
    return len(processor.tokenizer(text)["input_ids"]) + num_images * (width * height // 28**2 - 1)


if __name__ == "__main__":
    main()
