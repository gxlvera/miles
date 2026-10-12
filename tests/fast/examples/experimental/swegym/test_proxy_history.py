"""Claude round-trip regressions observed before the Proxy training run."""

import json
import os
from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from agent_proxy.continuous_token import create_builder
from agent_proxy.continuous_token.chat_template import apply_chat_template
from agent_proxy.protocol import Generation
from agent_proxy.trajectory import TrajectoryManager, context_key, message_fingerprint
from examples.experimental.swegym.claude_history import normalize_message

TOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}
}}}]
HISTORY = [{"role": "user", "content": "Inspect the project."}]
ASSISTANT = {"role": "assistant", "content": "\n\n", "reasoning_content": "Inspect the directory.",
             "tool_calls": [{"id": "one", "type": "function", "function": {
                 "name": "Bash", "arguments": {"command": "pwd"}
             }}]}
TAIL = {"role": "tool", "tool_call_id": "one", "content": "/testbed"}


@pytest.fixture(scope="module")
def tokenizer():
    path = os.getenv("AGENT_PROXY_TEST_TOKENIZER", "Qwen/Qwen3.5-4B")
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except OSError:
        pytest.skip("Local Qwen3.5 tokenizer is required")


def test_tool_argument_json_string_renders_like_mapping(tokenizer):
    replay = deepcopy(ASSISTANT)
    replay["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "pwd"})
    before = deepcopy(replay)
    expected = apply_chat_template(tokenizer, HISTORY + [ASSISTANT, TAIL], tools=TOOLS, add_generation_prompt=True)
    actual = apply_chat_template(tokenizer, HISTORY + [replay, TAIL], tools=TOOLS, add_generation_prompt=True)
    assert actual == expected
    assert replay == before


@pytest.mark.parametrize("content", [None, "", [{"type": "text", "text": "\n"}]])
def test_template_equivalent_claude_history_reuses_generated_tokens(tokenizer, content):
    replay = deepcopy(ASSISTANT)
    replay["content"] = content
    replay["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "pwd"})
    expected = apply_chat_template(tokenizer, HISTORY + [ASSISTANT, TAIL], tools=TOOLS, add_generation_prompt=True)
    assert apply_chat_template(tokenizer, HISTORY + [replay, TAIL], tools=TOOLS, add_generation_prompt=True) == expected
    builder = create_builder(tokenizer, model_type="qwen3_5")
    key = context_key(TOOLS, {})
    manager = TrajectoryManager(message_normalizer=normalize_message)
    first = manager.prepare(HISTORY, tools=TOOLS, builder=builder, context_key=key)
    full = apply_chat_template(tokenizer, HISTORY + [ASSISTANT], tools=TOOLS, add_generation_prompt=False)
    assert full[:len(first.input_ids)] == first.input_ids
    output = full[len(first.input_ids):]
    output = output[:output.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1]
    manager.commit(first, Generation(response={"id": "first"}, message=ASSISTANT, output_tokens=output,
                                    output_logprobs=[-0.5] * len(output), finish_reason="stop",
                                    meta_info={"weight_version": 1}, r3=None))
    second = manager.prepare(HISTORY + [replay, TAIL], tools=TOOLS, builder=builder, context_key=key)
    assert second.parent_id == 0
    assert second.input_ids == expected
    assert manager.nodes[0].output_tokens == output


def test_projection_preserves_real_content_and_argument_value_differences():
    original = deepcopy(ASSISTANT)
    projected = normalize_message(original)
    assert original == ASSISTANT
    changed = deepcopy(ASSISTANT)
    changed["tool_calls"][0]["function"]["arguments"]["command"] = " pwd "
    assert message_fingerprint(projected) != message_fingerprint(normalize_message(changed))
    changed = deepcopy(ASSISTANT)
    changed["content"] = "Do something else"
    assert message_fingerprint(projected) != message_fingerprint(normalize_message(changed))
