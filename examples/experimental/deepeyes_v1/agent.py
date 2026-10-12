"""DeepEyes v1 harness: seed-verl's scripts/deepeyes_v1 recipe (Qwen2.5-VL-7B-Instruct) as a Miles agent function.

An episode has at most 5 assistant turns:

- the prompt is the dataset's system and user messages. The tool schema comes from the recipe's chat
  template (chat_template.jinja, passed to Miles with --chat-template-path);
- each reply is read like verl's DeepEyesToolParser reads the raw generation: an ``<answer>...</answer>``
  ends the episode even next to a ``<tool_call>``, and of several ``<tool_call>`` blocks only the last runs;
- ``image_zoom_in_tool`` behaves like the recipe's DeepEyesZoomInTool. It crops the image the model sees,
  so the box is in that image's pixels, and answers with the crop and the official per-turn instruction.

SGLang runs without a tool-call parser: each reply comes back as raw text and is replayed as is, so the
next request extends exactly the tokens the policy generated.
"""

import asyncio
import json
import logging
import re

import httpx
import numpy as np
from examples.experimental.deepeyes.agent import image_block
from examples.experimental.deepeyes.reward import choice_judge
from examples.experimental.deepeyes_v1.reward import compute_score, extract_answer
from PIL import Image
from qwen_vl_utils import fetch_image

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."
# tool_config.yaml of the recipe.
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
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the "
                        "top-left corner and (x2, y2) is the bottom-right corner."
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
# Official PROMPT.USER_PROMPT_V2, the instruction that follows every crop.
USER_PROMPT_V2 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> "
    "(if tools needed)  <answer>...</answer> "
)
MAX_ASSISTANT_TURNS = 5
_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_METRICS = ("accuracy", "format_error", "tool_used", "judge_error", "num_tool_calls", "num_crops", "num_turns")


async def run(base_url: str, prompt: str, request_kwargs: dict, metadata: dict, **_) -> dict:
    view = await asyncio.to_thread(load_view, metadata["image"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": await asyncio.to_thread(user_content, prompt, view)},
    ]
    num_tool_calls = num_crops = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:
        for turn in range(MAX_ASSISTANT_TURNS):
            response = await client.post(
                f"{base_url}/v1/chat/completions", json={**request_kwargs, "messages": messages, "tools": [ZOOM_TOOL]}
            )
            if response.status_code != 200:
                # e.g. the context outgrew the budget: score what the policy produced so far.
                logger.warning("DeepEyes chat request failed (%d): %s", response.status_code, response.text[:500])
                break
            choice = response.json()["choices"][0]
            text = choice["message"]["content"] or ""
            messages.append({"role": "assistant", "content": text})
            call = parse_tool_call(text) if choice["finish_reason"] != "length" else None
            # Like verl's ToolAgentLoop, the last allowed turn ends the episode without running its tool.
            if call is None or turn == MAX_ASSISTANT_TURNS - 1:
                break
            tool_message = await asyncio.to_thread(zoom, view, call)
            messages.append(tool_message)
            num_tool_calls += 1
            num_crops += isinstance(tool_message["content"], list)

    solution = solution_text(messages[2:])
    reward, info = await compute_score(solution, ground_truth=metadata["answer"], question=metadata["question"])
    if metadata.get("reward") == "accuracy":  # multiple choice (V* Bench): the picked letter
        info["accuracy"] = float(choice_judge(metadata["answer"], extract_answer(solution), metadata["options"]))
        reward = info["accuracy"]
    num_turns = sum(message["role"] == "assistant" for message in messages)
    return {"reward": reward, **info, "num_tool_calls": num_tool_calls, "num_crops": num_crops, "num_turns": num_turns}


def load_view(path: str) -> Image.Image:
    """The image as the model sees it: qwen_vl_utils.fetch_image, as verl's process_image does."""
    return fetch_image({"image": Image.open(path)})


def user_content(prompt: str, image: Image.Image) -> list[dict]:
    return [
        image_block(image) if part == "<image>" else {"type": "text", "text": part} for part in prompt_parts(prompt)
    ]


def prompt_parts(prompt: str) -> list[str]:
    """The prompt split at ``<image>`` like verl's RLHFDataset; a prompt without the tag gets the image first."""
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return [part for part in re.split(r"(<image>)", prompt) if part]


def parse_tool_call(text: str) -> dict | None:
    """The call verl's DeepEyesToolParser runs for this reply, or None when the episode ends."""
    if _ANSWER.search(text):
        return None
    matches = _TOOL_CALL.findall(text)
    if not matches:
        return None
    try:
        call = json.loads(matches[-1])
        return {"name": call["name"], "arguments": call["arguments"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def zoom(view: Image.Image, call: dict) -> dict:
    """DeepEyesZoomInTool: clamp the box to the image, reject a side <= 30 px or an aspect ratio > 100,
    and answer with the crop, snapped to the 28 px grid by fetch_image, and USER_PROMPT_V2."""
    try:
        if call["name"] != ZOOM_TOOL["function"]["name"]:
            raise KeyError(call["name"])
        bbox = call["arguments"].get("bbox_2d")
        if not bbox or len(bbox) != 4:
            return _tool_text("Error: bbox_2d parameter is missing or not a list of 4 numbers.")
        box = crop_box([float(value) for value in bbox], *view.size)
        if box is None:
            return _tool_text("Error: ZOOM IN ARGUMENTS ARE INVALID")
        crop = fetch_image({"image": view.crop(box)})
    except Exception as exc:  # verl's ToolAgentLoop answers any tool failure with its text
        return _tool_text(f"Error when executing tool: {exc}")
    return {"role": "tool", "content": [image_block(crop), {"type": "text", "text": USER_PROMPT_V2}]}


def crop_box(bbox: list[float], width: int, height: int) -> list[float] | None:
    left, top = max(0.0, bbox[0]), max(0.0, bbox[1])
    right, bottom = min(float(width), bbox[2]), min(float(height), bbox[3])
    if not (left < right and top < bottom):
        return None
    box_width, box_height = right - left, bottom - top
    if max(box_width, box_height) / min(box_width, box_height) > 100 or min(box_width, box_height) <= 30:
        return None
    return [left, top, right, bottom]


def solution_text(messages: list[dict]) -> str:
    """The episode's response as verl's reward manager decodes it (special tokens skipped)."""
    parts = []
    for message in messages:
        if message["role"] == "assistant":
            parts.append(message["content"])
        else:
            content = message["content"]
            text = content if isinstance(content, str) else content[-1]["text"]
            parts.append(f"\nuser\n<tool_response>\n{text}\n</tool_response>\nassistant\n")
    return "".join(parts)


def _tool_text(text: str) -> dict:
    return {"role": "tool", "content": text}


async def score_failed_episodes(args, samples, **kwargs) -> list[float]:
    """Reward for samples whose agent function raised (Miles calls the reward model only for those): 0."""
    if samples:
        logger.warning("%d DeepEyes samples have no reward because their agent function failed", len(samples))
    return [0.0] * len(samples)


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Episode metrics (judge accuracy, format errors, tool use) next to Miles' default rollout metrics."""
    metrics = {f"rollout/deepeyes/{key}": value for key, value in _episode_metrics(samples).items()}
    tracking.log(args, {**metrics, "rollout/step": compute_rollout_step(args, rollout_id)}, step_key="rollout/step")
    return False


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    """The same per eval set, and the reward per data source (V* and ThinkLite in the held-out split)."""
    metrics = {}
    for name, result in data.items():
        samples = result.get("samples") or []
        metrics |= {f"eval/{name}/deepeyes/{key}": value for key, value in _episode_metrics(samples).items()}
        sources = {sample.metadata.get("data_source") for sample in samples} - {None}
        for source in sorted(sources) if len(sources) > 1 else []:
            rewards = [s.reward or 0.0 for s in samples if s.metadata.get("data_source") == source]
            metrics[f"eval/{name}/by_source/{source}"] = float(np.mean(rewards))
    tracking.log(args, {**metrics, "eval/step": compute_rollout_step(args, rollout_id)}, step_key="eval/step")
    return False


def _episode_metrics(samples) -> dict[str, float]:
    metrics = {}
    for key in _METRICS:
        values = [sample.metadata[key] for sample in samples if key in sample.metadata]
        if values:
            metrics[key] = float(np.mean(values))
    return metrics
