"""The Qwen and GLM builders (which run inside SGLang) against full chat-template renders of the same conversation.

Each conversation only appends where its template is append-only, so the incremental prompt must
equal the full render: tool results after the last user message for Qwen3.5 (whose template drops
earlier thinking once a new user message arrives), and anything for Qwen3-4B-Instruct-2507.
"""

import json
import os

import pytest

from agent_server.sglang_patch.continuous_token import (
    GLMContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    create_builder,
)
from agent_server.sglang_patch.continuous_token.chat_template import apply_chat_template

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
        assert merge.token_ids == apply_chat_template(tokenizer, messages, add_generation_prompt=True, tools=TOOLS)
        tokens = merge.token_ids


def test_user_and_parallel_tool_appends_equal_full_render():
    """A customer-service dialogue: a user reply, two parallel tool results, a tool result and a user message together."""
    transformers = pytest.importorskip("transformers")
    name = os.environ.get("AGENT_SERVER_TEST_INSTRUCT_TOKENIZER", "Qwen/Qwen3-4B-Instruct-2507")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"tokenizer {name} is not available locally")

    def call(tool, call_id):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": json.dumps({"command": tool})},
        }

    messages = [
        {"role": "system", "content": "You are a retail agent."},
        {"role": "user", "content": "Cancel my order."},
    ]
    script = [
        ({"role": "assistant", "content": "Sure, what is your email?"}, [{"role": "user", "content": "a@b.com"}]),
        (
            {"role": "assistant", "content": "", "tool_calls": [call("find_user", "c1"), call("get_order", "c2")]},
            [
                {"role": "tool", "tool_call_id": "c1", "content": "u_1"},
                {"role": "tool", "tool_call_id": "c2", "content": "pending"},
            ],
        ),
        (
            {"role": "assistant", "content": "", "tool_calls": [call("cancel_order", "c3")]},
            [
                {"role": "tool", "tool_call_id": "c3", "content": "cancelled"},
                {"role": "user", "content": "And the refund?"},
            ],
        ),
    ]
    tokens = create_builder(tokenizer, model_type="qwen3").build_initial_tokens(messages, tools=TOOLS)
    for assistant, appended in script:
        tokens = tokens + _model_output(tokenizer, messages, assistant)
        previous = [*messages, assistant]
        messages = [*previous, *appended]
        merge = create_builder(tokenizer, model_type="qwen3").merge_context_tokens(
            previous, messages, tokens, tools=TOOLS
        )
        assert merge.token_ids == apply_chat_template(tokenizer, messages, add_generation_prompt=True, tools=TOOLS)
        tokens = merge.token_ids


@pytest.mark.parametrize(
    "name, model_type, template_kwargs, reasoning",
    [
        # clear_thinking=False keeps earlier thinking, so the template stays append-only across user messages.
        ("zai-org/GLM-4.7-Flash", "glm4_moe_lite", {"clear_thinking": False}, True),
        # GLM-4.5 drops earlier thinking once a user message arrives; without thinking it is append-only.
        ("zai-org/GLM-4.5-Air", "glm4_moe", {"enable_thinking": False}, False),
    ],
)
def test_glm_appends_equal_full_render(name, model_type, template_kwargs, reasoning):
    """GLM stops at <|observation|> or <|user|>; the junction keeps one of them, whichever the next message opens with."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"tokenizer {name} is not available locally")
    observation, user = tokenizer.convert_tokens_to_ids(["<|observation|>", "<|user|>"])

    def render(messages, add_generation_prompt):
        return apply_chat_template(
            tokenizer, messages, add_generation_prompt=add_generation_prompt, tools=TOOLS, **template_kwargs
        )

    def model_output(history, assistant):
        """The rendered assistant turn plus the stop token GLM generates after it."""
        prompt = render(history, True)
        full = render([*history, assistant], False)
        assert full[: len(prompt)] == prompt
        return full[len(prompt) :] + [observation if assistant.get("tool_calls") else user]

    def assistant(content, call_id=None):
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = f"I should answer: {content or call_id}."
        if call_id:
            message["tool_calls"] = [
                {"id": call_id, "type": "function", "function": {"name": "env_step", "arguments": {"command": "ls"}}}
            ]
        return message

    script = [
        # stopped at <|observation|>, the tool result opens with it
        (assistant("", "c1"), [{"role": "tool", "tool_call_id": "c1", "content": "a.txt"}], 0),
        # stopped at <|user|>, the user message opens with it
        (assistant("There is a.txt."), [{"role": "user", "content": "Call it again."}], 0),
        # stopped at <|observation|>, but the user interrupts: the generated token is replaced
        (assistant("", "c2"), [{"role": "user", "content": "Never mind."}], 1),
        # a tool result and a user message together
        (
            assistant("", "c3"),
            [{"role": "tool", "tool_call_id": "c3", "content": "b.txt"}, {"role": "user", "content": "Done?"}],
            0,
        ),
    ]
    messages = [{"role": "system", "content": "You use a shell."}, {"role": "user", "content": "List files."}]
    tokens = create_builder(
        tokenizer, model_type=model_type, chat_template_kwargs=template_kwargs
    ).build_initial_tokens(messages, tools=TOOLS)
    assert tokens == render(messages, True)
    for reply, appended, removed in script:
        tokens = tokens + model_output(messages, reply)
        previous = [*messages, reply]
        messages = [*previous, *appended]
        builder = create_builder(tokenizer, model_type=model_type, chat_template_kwargs=template_kwargs)
        assert isinstance(builder, GLMContinuousTokenBuilder)
        merge = builder.merge_context_tokens(previous, messages, tokens, tools=TOOLS)
        assert merge.removed_prefix_token_count == removed
        assert merge.token_ids == render(messages, True)
        tokens = merge.token_ids
