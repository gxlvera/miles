"""Multimodal prompts against the HF processor run on the whole conversation, on CPU.

A mock trajectory with a new image in every turn (the user's picture, then one crop per zoom-in
tool call, then a follow-up question with another picture) is encoded two ways:

- reference: ``processor(text=chat_template(whole conversation), images=all images)``;
- agent server: ``TrajectoryManager``, one generation at a time. Each request goes through what
  patched SGLang does with it (``continue_prompt``: continuous token, groups with images encoded by
  the processor), the returned prompt edit is committed, and the exported image references go
  through the trainer's ``image_refs_to_train_inputs``.

The exported tokens must equal the reference input_ids, and both SGLang's and the trainer's
pixel_values and image_grid_thw must equal the reference ones.

With ``AGENT_SERVER_TEST_SGLANG_URL`` set to a patched SGLang serving ``AGENT_SERVER_TEST_PROCESSOR``'s
model, each request goes to that server instead, built and parsed by the agent server's backend module.
"""

import base64
import copy
import io
import json
import os

import numpy as np
import pytest
import torch

from agent_server.protocol import Generation, PromptEdit
from agent_server.sglang_patch.continuous_token import continue_prompt
from agent_server.sglang_patch.continuous_token.chat_template import normalize_token_ids
from agent_server.trajectory import TrajectoryManager, context_key
from miles.backends.training_utils.data.multimodal import image_refs_to_train_inputs
from miles.utils.processing_utils import load_image_ref

PROCESSOR = os.environ.get("AGENT_SERVER_TEST_PROCESSOR", "Qwen/Qwen3-VL-2B-Instruct")
SGLANG_URL = os.environ.get("AGENT_SERVER_TEST_SGLANG_URL")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "description": "Zoom in on a region of the image.",
            "parameters": {
                "type": "object",
                "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}}},
                "required": ["bbox_2d"],
            },
        },
    }
]
KEY = context_key(TOOLS, {})


@pytest.fixture(scope="module")
def processor():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    try:
        return transformers.AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    except OSError:
        pytest.skip(f"processor {PROCESSOR} is not available locally")


def _image_uri(width: int, height: int, seed: int) -> str:
    from PIL import Image

    pixels = (np.random.RandomState(seed).rand(height, width, 3) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _with_image(uri: str, text: str) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": text}]


def _without_images(conversation: list[dict]) -> list[dict]:
    """The same conversation as plain text, for the text-only path."""
    return [
        (
            {**message, "content": " ".join(block["text"] for block in message["content"] if block["type"] == "text")}
            if isinstance(message["content"], list)
            else message
        )
        for message in conversation
    ]


def _zoom(call_id: str, bbox: list[int]) -> dict:
    arguments = json.dumps({"bbox_2d": bbox})
    return {
        "role": "assistant",
        "content": "<think>Too small to read, zoom in.</think>",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "image_zoom_in_tool", "arguments": arguments}}
        ],
    }


def _conversation(crop_seed: int = 1) -> list[dict]:
    """Four generations, each answering a message that brings a new image."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _with_image(_image_uri(160, 120, seed=0), "What does the sign say?")},
        _zoom("call_0", [10, 10, 90, 70]),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": _with_image(_image_uri(80, 60, crop_seed), "Zoomed in."),
        },
        _zoom("call_1", [5, 5, 45, 35]),
        {"role": "tool", "tool_call_id": "call_1", "content": _with_image(_image_uri(40, 64, seed=2), "Zoomed in.")},
        {"role": "assistant", "content": "<think>Now it is legible.</think><answer>STOP</answer>"},
        {"role": "user", "content": _with_image(_image_uri(96, 200, seed=3), "And this one?")},
        {"role": "assistant", "content": "<answer>EXIT</answer>"},
    ]


def _image_refs(conversation: list[dict]) -> list[str]:
    return [
        block["image_url"]["url"]
        for message in conversation
        if isinstance(message["content"], list)
        for block in message["content"]
        if block["type"] == "image_url"
    ]


def _reference(processor, conversation: list[dict]) -> dict:
    """The whole conversation through the HF processor at once. The policy's last output ends at <|im_end|>."""
    images = [load_image_ref(ref) for ref in _image_refs(conversation)] or None
    # The processor rewrites image blocks in place.
    text = processor.apply_chat_template(copy.deepcopy(conversation), tools=TOOLS, tokenize=False)
    assert text.endswith("<|im_end|>\n")
    return processor(text=[text[: -len("\n")]], images=images, return_tensors="pt")


def _sglang(processor, prepared, messages: list[dict]) -> tuple[PromptEdit, dict | None]:
    """What patched SGLang does with an agent server request (sglang_patch/): the prompt edit it returns,
    and the image inputs it hands its model (only known when emulated here)."""
    if SGLANG_URL:
        return _served(prepared, messages), None
    continuation = continue_prompt(
        processor.tokenizer,
        processor,
        model_type="qwen3_vl",
        input_ids=prepared.parent_tokens,
        messages=messages,
        prefix_message_count=prepared.prefix_message_count,
        tools=TOOLS,
    )
    edit = PromptEdit(
        removed_prefix_token_count=len(prepared.parent_tokens) - continuation.prefix_len,
        input_delta=continuation.input_ids[continuation.prefix_len :],
        routed_experts_start_len=0,
    )
    return edit, continuation.image_inputs


def _served(prepared, messages: list[dict]) -> PromptEdit:
    import httpx

    from agent_server.backend import build_request, parse_generation

    body = build_request(
        {"messages": messages, "tools": TOOLS, "max_tokens": 1},
        parent_tokens=prepared.parent_tokens,
        prefix_message_count=prepared.prefix_message_count,
        chat_template_kwargs={},
        r3_row_start=None,
    )
    reply = httpx.post(f"{SGLANG_URL}/v1/chat/completions", json=body, timeout=300)
    assert reply.status_code == 200, reply.text
    generation = parse_generation(
        reply.json(), num_parent_tokens=len(prepared.parent_tokens), routing_replay=False, num_layers=None
    )
    return generation.prompt_edit


def _policy_output(tokenizer, conversation: list[dict], index: int) -> list[int]:
    """Tokens the policy generates for ``conversation[index]``: its rendering, ending at <|im_end|>."""
    prompt = normalize_token_ids(
        tokenizer.apply_chat_template(conversation[:index], tools=TOOLS, add_generation_prompt=True)
    )
    full = normalize_token_ids(tokenizer.apply_chat_template(conversation[: index + 1], tools=TOOLS))
    assert full[: len(prompt)] == prompt
    return full[len(prompt) : -1]  # drop the "\n" the template puts after <|im_end|>


def _play(manager: TrajectoryManager, processor, conversation: list[dict], *, start: int = 0) -> dict | None:
    """Serve the assistant messages of ``conversation`` from ``start`` on, as an agent replaying its history would.

    Returns the image inputs SGLang got for the last turn (None when a real server did the work).
    """
    image_inputs = None
    for index, message in enumerate(conversation):
        if message["role"] != "assistant" or index < start:
            continue
        prepared = manager.prepare(conversation[:index], context_key=KEY)
        edit, image_inputs = _sglang(processor, prepared, conversation[:index])
        output = _policy_output(processor.tokenizer, conversation, index)
        generation = Generation(
            response={"id": f"turn-{index}"},
            message=message,
            output_tokens=output,
            output_logprobs=[-0.1] * len(output),
            finish_reason="stop",
            meta_info={"weight_version": 1},
            prompt_edit=edit,
            r3=None,
        )
        manager.commit(prepared, generation)
    return image_inputs


def _assert_matches_reference(processor, sample, conversation: list[dict]) -> None:
    reference = _reference(processor, conversation)
    assert sample.tokens == reference["input_ids"][0].tolist()
    refs = _image_refs(conversation)
    assert sample.multimodal_inputs == {"images": refs}

    [train_inputs] = image_refs_to_train_inputs(processor, [sample.multimodal_inputs["images"]], [sample.tokens])
    assert torch.equal(train_inputs["image_grid_thw"], reference["image_grid_thw"])
    assert torch.equal(train_inputs["pixel_values"], reference["pixel_values"])


def test_turn_by_turn_expansion_matches_one_shot_processor(processor):
    conversation = _conversation()
    manager = TrajectoryManager()
    image_inputs = _play(manager, processor, conversation)

    # The last request holds all four images: what SGLang hands its model matches the one-shot processor.
    if image_inputs is not None:
        reference = _reference(processor, conversation)
        assert torch.equal(image_inputs["image_grid_thw"], reference["image_grid_thw"])
        assert torch.equal(image_inputs["pixel_values"], reference["pixel_values"])
    # Each replayed history matched the previous turn: one chain, one sample.
    assert [node.parent_id for node in manager.nodes] == [None, 0, 1, 2]
    assert [len(node.images) for node in manager.nodes] == [1, 1, 1, 1]
    [sample] = manager.leaf_samples(max_seq_len=None, tokenizer=processor.tokenizer)
    _assert_matches_reference(processor, sample, conversation)
    # Only policy output is trained on; every prompt token, image tokens included, has loss mask 0.
    assert sum(sample.loss_mask) == sum(len(node.output_tokens) for node in manager.nodes)


def test_branch_keeps_each_path_images(processor):
    """The agent retries the first tool call and gets a different crop: a branch after the first turn."""
    first, retry = _conversation(crop_seed=1), _conversation(crop_seed=11)
    manager = TrajectoryManager()
    _play(manager, processor, first)
    _play(manager, processor, retry, start=3)

    # The retry continues the first turn, whose expanded image is reused as is.
    assert manager.nodes[4].parent_id == 0
    samples = manager.leaf_samples(max_seq_len=None, tokenizer=processor.tokenizer)
    assert len(samples) == 2
    for sample, conversation in zip(samples, (first, retry), strict=True):
        _assert_matches_reference(processor, sample, conversation)


def test_text_only_turns_match_one_shot_render(processor):
    """The same conversation without images: continuous token alone, through the same SGLang path."""
    conversation = _without_images(_conversation())
    manager = TrajectoryManager()
    assert _play(manager, processor, conversation) is None
    [sample] = manager.leaf_samples(max_seq_len=None, tokenizer=processor.tokenizer)
    assert sample.tokens == _reference(processor, conversation)["input_ids"][0].tolist()
    assert sample.multimodal_inputs is None
