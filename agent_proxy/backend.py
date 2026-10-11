"""Calls to the Miles/SGLang inference router, and extraction of what the model really generated.

The request is OpenAI ``/v1/chat/completions`` with explicit ``input_ids``. SGLang generates from those ids and skips its own chat template, but still gets
``messages`` and ``tools`` because its reasoning and tool-call parsers build the reply message
from them. Token ids and logprobs come from ``meta_info.output_token_logprobs``, never from
re-tokenizing the reply text.
"""

from types import SimpleNamespace
from typing import Any

import httpx

from agent_proxy.protocol import Generation, ProxyError, R3Chunk
from miles.rollout.generate_utils.generate_endpoint_utils import get_routed_experts_from_response

# Client fields the proxy owns. The backend call is always non-streaming because the proxy needs
# the complete reply with meta_info before it can commit the turn.
_CLIENT_ONLY_KEYS = ("stream", "stream_options")


def build_request(request: dict[str, Any], *, input_ids: list[int], r3_row_start: int | None) -> dict[str, Any]:
    """The body for SGLang. ``r3_row_start`` is None when routing replay is off."""
    body = {key: value for key, value in request.items() if key not in _CLIENT_ONLY_KEYS}
    body.update(
        input_ids=input_ids,
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
    response = await client.post(
        f"{backend_url}/v1/chat/completions", json=body, headers={"X-SMG-Routing-Key": routing_key}
    )
    if response.status_code != 200:
        # e.g. 400 for a context that is too long. Nothing is committed; the agent may handle it.
        raise ProxyError(response.status_code, response.text)
    return response.json()


def parse_generation(
    response: dict[str, Any], *, prompt_len: int, r3_row_start: int | None, num_layers: int | None
) -> Generation:
    choice = response["choices"][0]
    if choice.get("finish_reason") == "abort":
        # Aborted for a weight update or a pause. There is nothing to train on; the agent sees an error.
        raise ProxyError(503, "generation aborted by the inference backend")
    meta_info = choice.get("meta_info") or {}
    token_logprobs = meta_info.get("output_token_logprobs")
    if token_logprobs is None or len(token_logprobs) != meta_info.get("completion_tokens"):
        raise ProxyError(
            502,
            "backend reply needs meta_info.output_token_logprobs with one entry per completion token "
            "(a Miles SGLang build, called with logprobs=True and return_meta_info=True)",
        )
    output_tokens = [entry[1] for entry in token_logprobs]
    output_logprobs = [entry[0] for entry in token_logprobs]

    r3 = None
    if r3_row_start is not None:
        # The backend returns rows up to the last input position of the final sequence.
        num_rows = prompt_len + len(output_tokens) - 1 - r3_row_start
        values = get_routed_experts_from_response(SimpleNamespace(num_layers=num_layers), choice, num_rows)
        if values is None:
            raise ProxyError(502, "routing replay is on but the backend reply has no meta_info.routed_experts")
        r3 = R3Chunk(row_start=r3_row_start, values=values)

    return Generation(
        response=response,
        message=choice["message"],
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        finish_reason=choice.get("finish_reason") or "",
        meta_info=meta_info,
        r3=r3,
    )
