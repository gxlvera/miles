"""Calls to the Miles/SGLang inference router, and extraction of what the model really generated.

The request is OpenAI ``/v1/chat/completions`` carrying the parent turn's tokens as ``input_ids``
and ``prefix_message_count``, the number of request messages those tokens cover. Patched SGLang
(``sglang_patch/``) renders the remaining messages onto them with continuous token, encoding
images with the model's processor, and reports in ``meta_info.prompt_edit`` how the prompt
extends the parent's tokens. Its reasoning and tool-call parsers build the reply message from
``messages`` and ``tools`` as usual. Token ids and logprobs come from
``meta_info.output_token_logprobs``, never from re-tokenizing the reply text.
"""

from types import SimpleNamespace
from typing import Any

import httpx
from agent_server.protocol import AgentServerError, Generation, PromptEdit, R3Chunk

from miles.rollout.generate_utils.generate_endpoint_utils import get_routed_experts_from_response

# Client fields the agent server owns. The backend call is always non-streaming because the agent server needs
# the complete reply with meta_info before it can commit the turn.
_CLIENT_ONLY_KEYS = ("stream", "stream_options")


def build_request(
    request: dict[str, Any],
    *,
    parent_tokens: list[int],
    prefix_message_count: int,
    chat_template_kwargs: dict[str, Any],
    r3_row_start: int | None,
) -> dict[str, Any]:
    """The body for SGLang. ``r3_row_start`` is None when routing replay is off."""
    body = {key: value for key, value in request.items() if key not in _CLIENT_ONLY_KEYS}
    body.update(
        input_ids=parent_tokens,
        prefix_message_count=prefix_message_count,
        # The launch defaults merged with the client's, as the session's trie keys were built from.
        chat_template_kwargs=chat_template_kwargs,
        stream=False,
        # Needed on every turn: the output token ids and their logprobs are read from meta_info.
        logprobs=True,
        return_meta_info=True,
        # Stop-token text is trimmed from the reply content; the token ids still include it.
        no_stop_trim=False,
    )
    if r3_row_start is not None:
        body.update(return_routed_experts=True, routed_experts_start_len=r3_row_start)
    return body


async def generate(client: httpx.AsyncClient, backend_url: str, body: dict[str, Any], *, routing_key: str) -> dict:
    # Session affinity: one session's turns land on one engine, which keeps its prefix cache warm.
    try:
        response = await client.post(
            f"{backend_url}/v1/chat/completions", json=body, headers={"X-SMG-Routing-Key": routing_key}
        )
    except httpx.TransportError as exc:
        raise AgentServerError(502, f"inference backend unreachable: {exc!r}") from exc
    if response.status_code != 200:
        # e.g. 400 for a context that is too long. Nothing is committed; the agent may handle it.
        raise AgentServerError(response.status_code, response.text)
    return response.json()


def parse_generation(
    response: dict[str, Any], *, num_parent_tokens: int, routing_replay: bool, num_layers: int | None
) -> Generation:
    choice = response["choices"][0]
    if choice.get("finish_reason") == "abort":
        # Aborted for a weight update or a pause. There is nothing to train on; the agent sees an error.
        raise AgentServerError(503, "generation aborted by the inference backend")
    meta_info = choice.get("meta_info") or {}
    token_logprobs = meta_info.get("output_token_logprobs")
    if token_logprobs is None or len(token_logprobs) != meta_info.get("completion_tokens"):
        raise AgentServerError(
            502,
            "backend reply needs meta_info.output_token_logprobs with one entry per completion token "
            "(a Miles SGLang build, called with logprobs=True and return_meta_info=True)",
        )
    output_tokens = [entry[1] for entry in token_logprobs]
    output_logprobs = [entry[0] for entry in token_logprobs]
    if meta_info.get("prompt_edit") is None:
        # Without it SGLang generated from the parent's tokens alone, not from this request's prompt.
        raise AgentServerError(
            502,
            "backend reply has no meta_info.prompt_edit: SGLang needs agent_server/sglang_patch, and the router in "
            "front of it must forward prefix_message_count",
        )
    prompt_edit = PromptEdit(**meta_info["prompt_edit"])
    prompt_len = num_parent_tokens - prompt_edit.removed_prefix_token_count + len(prompt_edit.input_delta)

    r3 = None
    if routing_replay:
        # The backend returns rows from routed_experts_start_len up to the last input position of the sequence.
        row_start = prompt_edit.routed_experts_start_len
        num_rows = prompt_len + len(output_tokens) - 1 - row_start
        values = get_routed_experts_from_response(SimpleNamespace(num_layers=num_layers), choice, num_rows)
        if values is None:
            raise AgentServerError(502, "routing replay is on but the backend reply has no meta_info.routed_experts")
        r3 = R3Chunk(row_start=row_start, values=values)

    return Generation(
        response=response,
        message=choice["message"],
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        finish_reason=choice.get("finish_reason") or "",
        meta_info=meta_info,
        prompt_edit=prompt_edit,
        r3=r3,
    )
