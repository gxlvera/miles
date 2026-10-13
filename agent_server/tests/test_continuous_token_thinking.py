"""Continuous token keeps a Qwen3 thinking model's earlier thinking with the model's own template.

Qwen3's template drops the thinking of earlier turns once a new user message arrives. Continuous token
never re-renders the history: it appends what each new message group renders to inside a synthetic
context. So with the native template the prompt still holds every turn's ``<think>`` block, exactly as
Miles' ``qwen3_fixed.jinja`` with ``clear_thinking: false`` renders the whole conversation; that render
is the reference here, not something the run uses.
"""

import json
import os
from pathlib import Path

import pytest

from agent_server.sglang_patch.continuous_token import create_builder
from agent_server.sglang_patch.continuous_token.chat_template import apply_chat_template

FIXED_TEMPLATE = Path(__file__).resolve().parents[2] / "miles/utils/chat_template_utils/templates/qwen3_fixed.jinja"
TEMPLATE_KWARGS = {"clear_thinking": False}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} of a reservation.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        },
    }
    for name in ("get_reservation", "cancel_reservation")
]


def test_customer_service_dialogue_with_thinking_equals_full_render():
    """User replies, parallel tool results, a tool result next to a user message, an empty thinking block."""
    transformers = pytest.importorskip("transformers")
    name = os.environ.get("AGENT_SERVER_TEST_THINKING_TOKENIZER", "Qwen/Qwen3-1.7B")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        reference = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"tokenizer {name} is not available locally")
    reference.chat_template = FIXED_TEMPLATE.read_text()

    def render(messages, add_generation_prompt):
        """The whole conversation with every turn's thinking kept."""
        return apply_chat_template(
            reference, messages, add_generation_prompt=add_generation_prompt, tools=TOOLS, **TEMPLATE_KWARGS
        )

    def model_output(history, assistant):
        """What the model generated for ``assistant``: its rendered turn up to and including <|im_end|>."""
        prompt = render(history, True)
        full = render([*history, assistant], False)
        assert full[: len(prompt)] == prompt
        output = full[len(prompt) :]
        return output[: output.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1]

    def assistant(reasoning, content="", calls=()):
        message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if calls:
            message["tool_calls"] = [
                {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps({"id": "R1"})}}
                for tool, call_id in calls
            ]
        return message

    script = [
        (assistant("The user wants a cancellation.", "Which reservation?"), [{"role": "user", "content": "R1."}]),
        (
            assistant("Look it up twice.", calls=[("get_reservation", "c1"), ("get_reservation", "c2")]),
            [
                {"role": "tool", "tool_call_id": "c1", "content": "R1: basic economy"},
                {"role": "tool", "tool_call_id": "c2", "content": "R1: basic economy"},
            ],
        ),
        (
            assistant("Policy allows it.", calls=[("cancel_reservation", "c3")]),
            [
                {"role": "tool", "tool_call_id": "c3", "content": "cancelled"},
                {"role": "user", "content": "And the refund?"},
            ],
        ),
        (assistant("", "It goes to your card."), [{"role": "user", "content": "Thanks."}]),
    ]
    messages = [{"role": "system", "content": "You are an airline agent."}, {"role": "user", "content": "Cancel."}]
    # The builder uses the checkpoint's own template, as the run does.
    tokens = create_builder(tokenizer, model_type="qwen3").build_initial_tokens(messages, tools=TOOLS)
    assert tokens == render(messages, True)
    for reply, appended in script:
        tokens = tokens + model_output(messages, reply)
        previous = [*messages, reply]
        messages = [*previous, *appended]
        merge = create_builder(tokenizer, model_type="qwen3").merge_context_tokens(
            previous, messages, tokens, tools=TOOLS
        )
        assert merge.token_ids == render(messages, True)
        tokens = merge.token_ids
    # Every turn's thinking is still in the prompt, while a full render with the native template drops it.
    thoughts = ("The user wants a cancellation.", "Look it up twice.", "Policy allows it.")
    assert all(thought in tokenizer.decode(tokens) for thought in thoughts)
    native = tokenizer.decode(apply_chat_template(tokenizer, messages, add_generation_prompt=True, tools=TOOLS))
    assert not any(thought in native for thought in thoughts)
