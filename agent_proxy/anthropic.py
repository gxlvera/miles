"""Anthropic Messages API (``/v1/messages``) for agents such as Claude Code.

SGLang's converter turns the request into an OpenAI chat request, the replica serves it through
its normal chat path, and the reply is converted back. Thinking blocks in the replayed history are
mapped back to ``reasoning_content``, so a replayed assistant turn matches the stored one and the
conversation keeps extending the same branch. ``stream: true`` is served as one burst of SSE
events after generation finishes. The converters only exist in the sglang-miles build; without
them the route answers 501.
"""

import json
from typing import Any

from agent_proxy.protocol import ProxyError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse
from sglang.srt.parser.template_detection import detect_inline_system_support
from starlette.responses import Response

from miles.rollout.session.anthropic_adapter import (
    _anthropic_error_response,
    _anthropic_sse_body,
    _anthropic_wire_json,
    _parse_anthropic_request,
    _restore_anthropic_reasoning_history,
    _strip_anthropic_reasoning_history,
    _validate_anthropic_features,
    anthropic_adapter_available,
)
from miles.rollout.session.core import JSON_MEDIA_TYPE, _client_response_body
from miles.rollout.session.request_args import ClientResponseIntent

try:
    from sglang.srt.entrypoints.anthropic import utils as anthropic_utils
    from sglang.srt.entrypoints.anthropic.serving import convert_response, convert_to_chat_completion_request
except ImportError:
    anthropic_utils = convert_response = convert_to_chat_completion_request = None


def add_anthropic_route(app: FastAPI, replica: Any) -> None:
    # Templates that cannot render a system message mid-conversation get it merged into the first one.
    merge_inline_system = not detect_inline_system_support(getattr(replica.tokenizer, "chat_template", None))

    @app.post("/sessions/{session_id}/v1/messages")
    async def anthropic_messages(session_id: str, request: Request):
        if convert_to_chat_completion_request is None or not anthropic_adapter_available():
            return JSONResponse(
                status_code=501,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": "The installed SGLang has no Anthropic converter"},
                },
            )
        try:
            anthropic_request = _parse_anthropic_request(await request.body())
            _validate_anthropic_features(anthropic_request)
            conversion_request, reasoning_history = _strip_anthropic_reasoning_history(anthropic_request)
            openai_request = convert_to_chat_completion_request(
                conversion_request, merge_inline_system=merge_inline_system
            )
            # Only what the client actually sent, so equivalent Anthropic and OpenAI requests look the same.
            openai_body = openai_request.model_dump(mode="json", exclude_none=True, exclude_unset=True, by_alias=True)
            openai_body.pop("stream", None)
            openai_body.pop("stream_options", None)
            _restore_anthropic_reasoning_history(openai_body, reasoning_history)
        except ValueError as exc:
            return _anthropic_error_response(400, _error_body(str(exc)))

        try:
            reply = await replica.chat(session_id, openai_body)
        except ProxyError as exc:
            return _anthropic_error_response(exc.status_code, _error_body(exc.message))

        openai_response = ChatCompletionResponse.model_validate(
            _client_response_body(reply, ClientResponseIntent(stream=False, logprobs=False))
        )
        if anthropic_request.stream:
            events = anthropic_utils.to_anthropic_fake_sse_events(
                openai_response, model=anthropic_request.model, id_factory=lambda: openai_response.id
            )
            return Response(
                content=_anthropic_sse_body(events),
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                media_type="text/event-stream",
            )
        envelope = convert_response(openai_response).model_copy(update={"id": openai_response.id})
        return Response(content=_anthropic_wire_json(envelope), media_type=JSON_MEDIA_TYPE)


def _error_body(message: str) -> bytes:
    return json.dumps({"error": message}).encode()
