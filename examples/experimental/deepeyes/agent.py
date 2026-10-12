"""DeepEyes harness: answer a question about an image, zooming in with ``image_zoom_in_tool`` when needed.

Port of verl's DeepEyes recipe (``recipe/deepeyes``: system prompt, tool, at most 5 assistant turns,
reward) as a Miles agent function behind agent_server. The harness speaks OpenAI chat to
``{base_url}/v1/chat/completions``, sends every image (the question's and each crop) as a PNG data
URI, and executes the tool itself by cropping the original image. The agent server turns each request into
tokens and image references; the agent never sees token ids.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
from math import ceil, floor

import httpx
from examples.experimental.deepeyes.reward import compute_score
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. You can call functions to assist with the user query. "
    "Important: You must call only one function at a time. After each function call, "
    "wait for the execution result before making the next function call if needed."
)
ZOOM_TOOL = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
            "optional object label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2] on a 0-1000 grid over the "
                        "image, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box (optional).",
                },
            },
            "required": ["bbox_2d"],
        },
    },
}
MAX_ASSISTANT_TURNS = int(os.environ.get("DEEPEYES_MAX_TURNS", "5"))
MIN_CROP = 28  # the tool's minimum crop side in pixels, as in verl's ImageZoomInTool
# Qwen3-VL grounds on a relative 0-1000 grid (Qwen2.5-VL, which the recipe used, gave pixels). A probe of
# the policy's boxes (probe_bbox.py) confirmed it: half of them overshoot small images in pixel terms,
# and none passes 1000 on large ones.
BBOX_GRID = 1000
MAX_PIXELS = int(os.environ.get("DEEPEYES_MAX_PIXELS", str(1024 * 1024)))  # bounds the tokens per image sent


async def run(base_url: str, prompt: str, request_kwargs: dict, metadata: dict, **_) -> dict:
    # The model sees a view of at most MAX_PIXELS; zoom-ins crop the original. Training images are stored
    # within the bound already, so for them the view is the original. Image work runs off the event loop.
    original = await asyncio.to_thread(lambda: Image.open(metadata["image"]).convert("RGB"))
    view = await asyncio.to_thread(_fit, original)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [await asyncio.to_thread(_image_block, view), {"type": "text", "text": prompt}]},
    ]
    num_tool_calls = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:
        for _ in range(MAX_ASSISTANT_TURNS):
            response = await client.post(
                f"{base_url}/v1/chat/completions", json={**request_kwargs, "messages": messages, "tools": [ZOOM_TOOL]}
            )
            if response.status_code != 200:
                # e.g. the context outgrew the budget: score what the policy produced so far.
                logger.warning("DeepEyes chat request failed (%d): %s", response.status_code, response.text[:500])
                break
            choice = response.json()["choices"][0]
            message = choice["message"]
            # Replayed exactly as returned, so the next request extends this turn in the agent server.
            messages.append(
                {
                    key: message[key]
                    for key in ("role", "content", "reasoning_content", "tool_calls")
                    if message.get(key)
                }
            )
            if choice["finish_reason"] == "length" or not message.get("tool_calls"):
                break
            num_tool_calls += 1
            messages.append(await asyncio.to_thread(_zoom, original, message["tool_calls"][0]))

    assistant_texts = [message.get("content") or "" for message in messages if message["role"] == "assistant"]
    reward, info = await compute_score(
        assistant_texts,
        used_tool=num_tool_calls > 0,
        question=metadata["question"],
        answer=metadata["answer"],
        options=metadata.get("options"),
    )
    if metadata.get("reward") == "accuracy":  # eval sets such as V* Bench report plain accuracy
        reward = info["accuracy"]
    return {"reward": reward, "num_tool_calls": num_tool_calls, "num_turns": len(assistant_texts), **info}


async def score_failed_episodes(args, samples, **kwargs) -> list[float]:
    """Reward for samples the agent returned none for, i.e. episodes whose agent function raised.

    Miles calls the reward model only for those; the others got their reward from the agent through the
    agent server's postprocessor. They score 0, like an episode that gave no answer.
    """
    if samples:
        logger.warning("%d DeepEyes samples have no reward because their agent function failed", len(samples))
    return [0.0] * len(samples)


def _zoom(original: Image.Image, tool_call: dict) -> dict:
    """Run one ``image_zoom_in_tool`` call; failures come back to the policy as text.

    The box is on the 0-1000 grid, so it maps onto the original whatever view the model saw.
    """
    try:
        arguments = json.loads(tool_call["function"]["arguments"] or "{}")
        bbox = arguments["bbox_2d"]
        scale = (original.width / BBOX_GRID, original.height / BBOX_GRID) * 2
        box = _crop_box([float(value) * factor for value, factor in zip(bbox, scale, strict=True)], *original.size)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        box, bbox, arguments = None, None, {}
    if box is None:
        content = (
            f"Error: the bounding box {bbox} is invalid or gives a crop smaller than {MIN_CROP}x{MIN_CROP}."
            if bbox is not None
            else "Error: bbox_2d parameter is missing or not a list of 4 numbers."
        )
    else:
        text = f"Zoomed in on the image to the region {bbox}"
        if arguments.get("label"):
            text += f" with label {arguments['label']}"
        content = [_image_block(_fit(original.crop(box))), {"type": "text", "text": text + "."}]
    return {"role": "tool", "tool_call_id": tool_call.get("id"), "content": content}


def _crop_box(bbox: list[float], width: int, height: int) -> list[float] | None:
    """Clamp the box to the image and grow it around its center to at least MIN_CROP per side."""
    if len(bbox) != 4:
        return None
    left, top = max(0.0, bbox[0]), max(0.0, bbox[1])
    right, bottom = min(float(width), bbox[2]), min(float(height), bbox[3])
    if right <= left or bottom <= top:
        return None
    box_width, box_height = right - left, bottom - top
    if min(box_width, box_height) < MIN_CROP:
        scale = MIN_CROP / min(box_width, box_height)
        box_width, box_height = min(box_width * scale, width), min(box_height * scale, height)
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        left = min(max(0.0, center_x - box_width / 2), width - box_width)
        top = min(max(0.0, center_y - box_height / 2), height - box_height)
        left, top, right, bottom = floor(left), floor(top), ceil(left + box_width), ceil(top + box_height)
    if min(right - left, bottom - top) < MIN_CROP:
        return None
    return [left, top, right, bottom]


def _fit(image: Image.Image) -> Image.Image:
    """The image downscaled to at most MAX_PIXELS (as prepare_data.py does for training images)."""
    width, height = image.size
    if width * height <= MAX_PIXELS:
        return image
    scale = math.sqrt(MAX_PIXELS / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def _image_block(image: Image.Image) -> dict:
    # PNG: SGLang and the trainer must decode identical pixels, and lossless PNG goes through PIL on both.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()},
    }
