"""Turn-by-turn prompts of DeepEyes v1 episodes against the HF processor run on the whole episode (CPU).

Real rows (the held-out split and V* Bench), the v1 harness (view, raw ``<tool_call>`` replies,
DeepEyesZoomInTool crops, USER_PROMPT_V2), the recipe's chat template and the Qwen2.5-VL processor.
Each generation goes through what patched SGLang does with it (``continue_prompt``) and is committed
to a TrajectoryManager; the exported sample must equal ``processor(template(episode), images)``, and
the trainer's pixel_values (from the image references) the processor's.

Example:
  python agent_server/examples/deepeyes_v1/check_tokens.py --model /vol/models/Qwen2.5-VL-7B-Instruct \\
      --data /vol/data/deepeyes_v1/val.jsonl /vol/data/vstar_bench/test.jsonl
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from agent_server.examples.deepeyes_v1.agent import (
    SYSTEM_PROMPT,
    ZOOM_TOOL,
    load_view,
    parse_tool_call,
    user_content,
    zoom,
)
from agent_server.protocol import Generation, PromptEdit
from agent_server.sglang_patch.continuous_token import continue_prompt
from agent_server.sglang_patch.continuous_token.chat_template import normalize_token_ids
from agent_server.trajectory import TrajectoryManager, context_key
from transformers import AutoProcessor

from miles.backends.training_utils.data.multimodal import image_refs_to_train_inputs
from miles.utils.processing_utils import load_image_ref

TEMPLATE = Path(__file__).resolve().parent / "chat_template.jinja"
TOOLS = [ZOOM_TOOL]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--rows-per-file", type=int, default=6)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    processor.chat_template = processor.tokenizer.chat_template = TEMPLATE.read_text()
    episodes = matched = 0
    for path in args.data:
        for line in open(path).readlines()[: args.rows_per_file]:
            row = json.loads(line)
            episode = _episode(row)
            episodes += 1
            matched += _check(processor, episode)
    print(f"episodes={episodes} matching_one_shot_processor={matched}")
    assert matched == episodes


def _episode(row: dict) -> list[dict]:
    """Two zoom-ins (the second one invalid when the crop is small) and an answer, as the harness builds them."""
    view = load_view(row["metadata"]["image"])
    width, height = view.size
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(row["prompt"], view)},
    ]
    for bbox in ([width // 4, height // 4, 3 * width // 4, 3 * height // 4], [0, 0, 20, 20]):
        reply = (
            "<think>The detail is small; zoom in.</think>\n<tool_call>\n"
            + json.dumps({"name": "image_zoom_in_tool", "arguments": {"bbox_2d": bbox}})
            + "\n</tool_call>"
        )
        messages.append({"role": "assistant", "content": reply})
        messages.append(zoom(view, parse_tool_call(reply)))
    messages.append({"role": "assistant", "content": "<think>Now it is clear.</think>\n<answer>yes</answer>"})
    return messages


def _check(processor, episode: list[dict]) -> bool:
    manager = TrajectoryManager()
    key = context_key(TOOLS, {})
    for index, message in enumerate(episode):
        if message["role"] != "assistant":
            continue
        prepared = manager.prepare(episode[:index], context_key=key)
        continuation = continue_prompt(
            processor.tokenizer,
            processor,
            model_type="qwen2_5_vl",
            input_ids=prepared.parent_tokens,
            messages=episode[:index],
            prefix_message_count=prepared.prefix_message_count,
            tools=TOOLS,
        )
        edit = PromptEdit(
            removed_prefix_token_count=len(prepared.parent_tokens) - continuation.prefix_len,
            input_delta=continuation.input_ids[continuation.prefix_len :],
            routed_experts_start_len=0,
        )
        output = _policy_output(processor.tokenizer, episode, index)
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

    [sample] = manager.leaf_samples(max_seq_len=None, tokenizer=processor.tokenizer)
    refs = sample.multimodal_inputs["images"]
    text = processor.apply_chat_template(copy.deepcopy(episode), tools=TOOLS, tokenize=False)
    reference = processor(text=[text[: -len("\n")]], images=[load_image_ref(ref) for ref in refs], return_tensors="pt")
    [train_inputs] = image_refs_to_train_inputs(processor, [refs], [sample.tokens])
    chain = [node.parent_id for node in manager.nodes] == [None, *range(len(manager.nodes) - 1)]
    ok = (
        chain
        and sample.tokens == reference["input_ids"][0].tolist()
        and torch.equal(train_inputs["image_grid_thw"], reference["image_grid_thw"])
        and torch.equal(train_inputs["pixel_values"], reference["pixel_values"])
    )
    print(f"images={len(refs)} grids={reference['image_grid_thw'].tolist()} tokens={len(sample.tokens)} ok={ok}")
    return ok


def _policy_output(tokenizer, episode: list[dict], index: int) -> list[int]:
    """Tokens the policy generates for ``episode[index]``: its rendering, ending at <|im_end|>."""
    prompt = normalize_token_ids(
        tokenizer.apply_chat_template(episode[:index], tools=TOOLS, add_generation_prompt=True)
    )
    full = normalize_token_ids(tokenizer.apply_chat_template(episode[: index + 1], tools=TOOLS))
    assert full[: len(prompt)] == prompt
    return full[len(prompt) : -1]


if __name__ == "__main__":
    main()
