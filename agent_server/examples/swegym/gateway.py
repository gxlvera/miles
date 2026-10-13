"""Expose only authenticated Anthropic requests to the agent server through a Modal TLS tunnel.

Session lifecycle, collection and the SGLang router remain private to the trainer.
"""

import hmac
import os
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

_ALLOWED = re.compile(
    r"^/(?:workers/(?P<worker>[0-9]+)/)?(?P<session>sessions/[A-Za-z0-9_-]+/v1/messages(?:/count_tokens)?)$"
)


def normalize_claude_request(payload: dict) -> dict:
    """Translate Claude-only hints into the text/tool subset served by Qwen.

    Claude effort does not control Qwen's server-side reasoning mode. OpenAI tool messages have no
    is_error flag, so preserve the failure explicitly in the visible tool text.
    Other output_config features remain subject to Miles' strict validation.
    """
    body = deepcopy(payload)
    config = body.get("output_config")
    if isinstance(config, dict) and set(config) == {"effort"}:
        body.pop("output_config")
    for message in body.get("messages", []):
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "tool_result" or block.get("is_error") is not True:
                continue
            block.pop("is_error")
            content = block.get("content", "")
            notice = "[Tool execution failed]\n"
            block["content"] = (
                notice + content if isinstance(content, str) else [{"type": "text", "text": notice}, *content]
            )
    return body


def create_app() -> FastAPI:
    token = Path(os.environ["SWEGYM_GATEWAY_TOKEN_FILE"]).read_text().strip()
    if len(token) < 32:
        raise ValueError("Gateway token must contain at least 32 characters")
    upstream = os.getenv("SWEGYM_AGENT_SERVER_URL", "http://127.0.0.1:30000")
    router = os.getenv("SWEGYM_ROUTER_INTERNAL_URL", "http://127.0.0.1:31000")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.post("/{path:path}")
    async def forward(path: str, request: Request):
        given = request.headers.get("x-api-key", "") or request.headers.get("authorization", "").removeprefix(
            "Bearer "
        )
        if not hmac.compare_digest(given, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        match = _ALLOWED.fullmatch("/" + path)
        if not match:
            return JSONResponse({"error": "not found"}, status_code=404)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "content-encoding", "authorization", "x-api-key"}
        }
        worker = int(match.group("worker") or 0)
        if worker >= int(os.getenv("SWEGYM_AGENT_SERVER_WORKERS", "1")):
            return JSONResponse({"error": "not found"}, status_code=404)
        internal = urlsplit(upstream)
        origin = urlunsplit((internal.scheme, f"{internal.hostname}:{(internal.port or 30000) + worker}", "", "", ""))
        destination = origin + "/" + match.group("session")
        if path.endswith("/count_tokens"):
            # Route this non-generating endpoint directly to the serving tokenizer.
            # Use the serving engine's tokenizer, without creating a trajectory.
            workers = await app.state.client.get(router + "/workers")
            workers.raise_for_status()
            urls = [worker["url"] for worker in workers.json()["workers"]]
            if not urls:
                return JSONResponse({"error": "inference is not ready"}, status_code=503)
            destination = urls[0].rstrip("/") + "/v1/messages/count_tokens"
        upstream_request = app.state.client.build_request(
            "POST",
            destination,
            json=normalize_claude_request(await request.json()),
            headers=headers,
        )
        response = await app.state.client.send(upstream_request, stream=True)

        async def body():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            body(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    return app


if __name__ == "__main__":
    import uvicorn
    from miles.utils.misc import get_current_node_ip

    os.environ.setdefault("SWEGYM_ROUTER_INTERNAL_URL", f"http://{get_current_node_ip()}:31000")
    uvicorn.run(create_app(), host="0.0.0.0", port=30080)
