"""Retries of the backend call: transport errors and 5xx replies are sent again, 4xx replies are not."""

import asyncio

import httpx
import pytest
from agent_server import backend
from agent_server.protocol import AgentServerError


def _call(replies: list, *, max_retries: int) -> tuple[dict | AgentServerError, int]:
    """Run ``generate`` against a backend that answers with ``replies`` in order; return the result and the call count."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        reply = replies[min(calls, len(replies) - 1)]
        calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await backend.generate(
                client, "http://router", {"messages": []}, routing_key="session", max_retries=max_retries
            )

    try:
        result = asyncio.run(run())
    except AgentServerError as exc:
        result = exc
    return result, calls


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(backend, "_RETRY_DELAY_SECONDS", 0.0)


def test_transport_error_and_5xx_are_retried():
    replies = [
        httpx.ConnectError("engine down"),
        httpx.Response(500, text="worker connection failed"),
        httpx.Response(200, json={"id": "ok"}),
    ]
    assert _call(replies, max_retries=3) == ({"id": "ok"}, 3)


def test_error_goes_to_the_agent_after_max_retries():
    error, calls = _call([httpx.Response(503, text="busy")], max_retries=3)
    assert isinstance(error, AgentServerError) and error.status_code == 503
    assert calls == 4


def test_4xx_is_not_retried():
    error, calls = _call([httpx.Response(400, text="context too long")], max_retries=3)
    assert isinstance(error, AgentServerError) and error.status_code == 400
    assert calls == 1
