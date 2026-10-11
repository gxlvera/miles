"""The ported Qwen builder against full chat-template renders of the same conversation.

The conversation only appends tool results after the last user message, where Qwen3.5's
template is append-only, so the incremental prompt must equal the full render.
"""

from agent_proxy.continuous_token import QwenContinuousTokenBuilder, create_builder
from agent_proxy.continuous_token.chat_template import apply_chat_template

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "env_step",
            "description": "Run one command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def _assistant(command, call_id):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": f"I should {command}.",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "env_step", "arguments": {"command": command}}}
        ],
    }


def _model_output(tokenizer, history, assistant):
    """What the model would have generated for ``assistant``: its rendered turn up to and including <|im_end|>."""
    prompt = apply_chat_template(tokenizer, history, add_generation_prompt=True, tools=TOOLS)
    full = apply_chat_template(tokenizer, [*history, assistant], add_generation_prompt=False, tools=TOOLS)
    assert full[: len(prompt)] == prompt
    output = full[len(prompt) :]
    end = output.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return output[: end + 1]


def test_incremental_prompt_equals_full_render(tokenizer):
    builder = create_builder(tokenizer, model_type="qwen3_5")
    assert isinstance(builder, QwenContinuousTokenBuilder)

    messages = [{"role": "system", "content": "You play a text game."}, {"role": "user", "content": "Find the mug."}]
    tokens = builder.build_initial_tokens(messages, tools=TOOLS)
    assert tokens == apply_chat_template(tokenizer, messages, add_generation_prompt=True, tools=TOOLS)

    for step, command in enumerate(["go to desk 1", "take mug 1 from desk 1"]):
        assistant = _assistant(command, f"call_{step}")
        tokens = tokens + _model_output(tokenizer, messages, assistant)  # stops at <|im_end|>, no "\n"
        previous = [*messages, assistant]
        messages = [*previous, {"role": "tool", "tool_call_id": f"call_{step}", "content": f"observation {step}"}]

        merge = builder.merge_context_tokens(previous, messages, tokens, tools=TOOLS)
        assert merge.removed_prefix_token_count == 0
        assert merge.inserted_token_ids == tokenizer.encode("\n", add_special_tokens=False)
        assert merge.token_ids == apply_chat_template(tokenizer, messages, add_generation_prompt=True, tools=TOOLS)
        tokens = merge.token_ids
