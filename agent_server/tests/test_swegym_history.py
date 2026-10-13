"""Claude Code round-trip regressions observed before the SWE-Gym training run."""

import json
from copy import deepcopy

import pytest

from agent_server.protocol import Generation, PromptEdit
from agent_server.sglang_patch.continuous_token import continue_prompt
from agent_server.sglang_patch.continuous_token.chat_template import apply_chat_template
from agent_server.trajectory import TrajectoryManager, context_key, message_fingerprint

TOOLS = [
    {
        "type": "function",
        "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
    }
]
HISTORY = [{"role": "user", "content": "Inspect the project."}]
ASSISTANT = {
    "role": "assistant",
    "content": "\n\n",
    "reasoning_content": "Inspect the directory.",
    "tool_calls": [{"id": "one", "type": "function", "function": {"name": "Bash", "arguments": {"command": "pwd"}}}],
}
TAIL = {"role": "tool", "tool_call_id": "one", "content": "/testbed"}


def _continue(tokenizer, input_ids, messages, prefix_message_count):
    """The prompt SGLang builds for the agent server (continuous token)."""
    return continue_prompt(
        tokenizer,
        None,
        model_type="qwen3_5",
        input_ids=input_ids,
        messages=messages,
        prefix_message_count=prefix_message_count,
        tools=TOOLS,
    )


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
    assert (
        apply_chat_template(tokenizer, HISTORY + [replay, TAIL], tools=TOOLS, add_generation_prompt=True) == expected
    )
    key = context_key(TOOLS, {})
    manager = TrajectoryManager()
    first = manager.prepare(HISTORY, context_key=key)
    prompt = _continue(tokenizer, [], HISTORY, 0).input_ids
    full = apply_chat_template(tokenizer, HISTORY + [ASSISTANT], tools=TOOLS, add_generation_prompt=False)
    assert full[: len(prompt)] == prompt
    output = full[len(prompt) :]
    output = output[: output.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1]
    manager.commit(
        first,
        Generation(
            response={"id": "first"},
            message=ASSISTANT,
            output_tokens=output,
            output_logprobs=[-0.5] * len(output),
            finish_reason="stop",
            meta_info={"weight_version": 1},
            prompt_edit=PromptEdit(0, prompt, 0),
            r3=None,
        ),
    )
    second = manager.prepare(HISTORY + [replay, TAIL], context_key=key)
    assert second.parent_id == 0
    continuation = _continue(tokenizer, second.parent_tokens, HISTORY + [replay, TAIL], second.prefix_message_count)
    assert continuation.input_ids == expected
    assert manager.nodes[0].output_tokens == output


def test_fingerprint_keeps_real_content_differences():
    changed = deepcopy(ASSISTANT)
    changed["content"] = "Do something else"
    assert message_fingerprint(ASSISTANT) != message_fingerprint(changed)
