"""Which coordinate frame does the policy use for ``image_zoom_in_tool`` boxes?

Sends the first turn of DeepEyes episodes (training rows and V* Bench rows, exactly as agent.py
builds them) to an OpenAI-compatible server and records every bbox the policy asks for, with the
size of the view it saw. Pixel boxes stay inside the view; Qwen3-VL's native grounding uses a
relative 0-1000 grid, whose boxes exceed small views and never pass 1000 on large ones.

Example:
  python examples/experimental/deepeyes/probe_bbox.py --url http://127.0.0.1:30000 \\
      --data /vol/data/deepeyes/train.jsonl /vol/data/vstar_bench/test.jsonl --out /vol/probe/bbox.json
"""

import argparse
import asyncio
import json

import httpx
from PIL import Image

from examples.experimental.deepeyes.agent import SYSTEM_PROMPT, ZOOM_TOOL, _fit, _image_block


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--rows-per-file", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for path in args.data for line in open(path).readlines()[: args.rows_per_file]]
    async with httpx.AsyncClient(timeout=600) as client:
        results = await asyncio.gather(*(_first_turn(client, args.url, row) for row in rows))
    calls = [call for result in results for call in result]
    with open(args.out, "w") as f:
        json.dump(calls, f)

    boxes = [call for call in calls if isinstance(call["bbox"], list) and len(call["bbox"]) == 4]
    outside = [c for c in boxes if c["bbox"][2] > c["view"][0] + 1 or c["bbox"][3] > c["view"][1] + 1]
    large_views = [c for c in boxes if max(c["view"]) > 1000]
    above_1000 = [c for c in large_views if max(c["bbox"]) > 1000]
    print(f"episodes={len(rows)} zoom_calls={len(calls)} with_4_numbers={len(boxes)}")
    print(f"boxes outside the view (impossible in pixels): {len(outside)}/{len(boxes)}")
    print(f"views larger than 1000 px: {len(large_views)}, boxes there with a coordinate > 1000: {len(above_1000)}")


async def _first_turn(client: httpx.AsyncClient, url: str, row: dict) -> list[dict]:
    view = _fit(Image.open(row["metadata"]["image"]).convert("RGB"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [_image_block(view), {"type": "text", "text": row["prompt"]}]},
    ]
    body = {"model": "policy", "messages": messages, "tools": [ZOOM_TOOL], "temperature": 1.0, "max_tokens": 1024}
    reply = (await client.post(f"{url}/v1/chat/completions", json=body)).json()
    calls = []
    for call in reply["choices"][0]["message"].get("tool_calls") or []:
        try:
            bbox = json.loads(call["function"]["arguments"]).get("bbox_2d")
        except (json.JSONDecodeError, AttributeError):
            bbox = None
        calls.append(
            {
                "image": row["metadata"]["image"],
                "question": row["metadata"]["question"],
                "view": view.size,
                "bbox": bbox,
            }
        )
    return calls


if __name__ == "__main__":
    asyncio.run(main())
