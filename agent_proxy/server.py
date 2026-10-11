"""Proxy replica: one process that owns a set of sessions.

A chat request goes prepare (pick the parent turn, render the prompt with continuous token)
-> generate on the inference backend (no session state is touched while waiting) -> commit
(store the new turn). Collect exports one Miles Sample per leaf, runs the configured picker and
postprocessor hooks, and encodes the result with Miles' samples codec.

    POST   /sessions                            create (called by the Router)
    POST   /sessions/{id}/v1/chat/completions   OpenAI chat, for the agent
    POST   /sessions/{id}/v1/messages           Anthropic messages, for agents such as Claude Code
    POST   /sessions/{id}/samples               export samples; carries the agent's reward/metadata
    DELETE /sessions/{id}                       drop everything the session holds
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import setproctitle
import uvicorn
from agent_proxy.anthropic import add_anthropic_route
from agent_proxy.backend import build_request, generate, parse_generation
from agent_proxy.continuous_token import ContinuousTokenBuilder, create_builder
from agent_proxy.protocol import ProxyError
from agent_proxy.trajectory import TrajectoryManager, context_key
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.responses import Response
from transformers import AutoConfig

from miles.rollout.session.core import JSON_MEDIA_TYPE, _client_response_body, _render_json, _response_to_stream_chunk
from miles.rollout.session.request_args import ClientResponseIntent
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, encode_samples
from miles.rollout.session.types import CreateSessionRequest
from miles.utils.function_registry import load_function
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.processing_utils import load_tokenizer
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


class ReplicaConfig(FrozenStrictBaseModel):
    host: str
    port: int
    router_url: str | None  # where to register on startup; None runs the replica standalone
    # The URL agents use when they cannot reach host:port directly, e.g. a tunnel to a sandbox
    # provider outside the cluster. The Router and the Miles driver keep using host:port.
    public_url: str | None = None
    backend_url: str  # the Miles/SGLang inference router
    hf_checkpoint: str
    chat_template_path: str | None = None
    apply_chat_template_kwargs: dict[str, Any] | None = None
    timeout: float | None = None
    use_rollout_routing_replay: bool = False
    # Every pause mode except "retract" keeps earlier routing rows valid, so the backend only has to
    # return the rows a turn adds.
    use_addition_r3: bool = True
    num_layers: int | None = None
    session_sample_picker_path: str = "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
    session_sample_postprocessor_path: str = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class Session:
    session_id: str
    evaluation: bool
    sampling_defaults: dict[str, Any]
    manager: TrajectoryManager = field(default_factory=TrajectoryManager)
    closed: bool = False


class Replica:
    def __init__(self, config: ReplicaConfig, *, tokenizer: Any, model_type: str | None, client: httpx.AsyncClient):
        self.config = config
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.client = client
        self.sessions: dict[str, Session] = {}
        self._builders: dict[str, ContinuousTokenBuilder] = {}
        self._sample_picker = load_function(config.session_sample_picker_path, sync_required=True)
        self._sample_postprocessor = load_function(config.session_sample_postprocessor_path, sync_required=True)

    def create_session(self, params: CreateSessionRequest) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = Session(
            session_id=session_id,
            evaluation=params.evaluation,
            sampling_defaults=params.model_dump(exclude={"evaluation"}, exclude_none=True),
        )
        return session_id

    def get_session(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise ProxyError(404, f"session not found: {session_id}")
        return session

    def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        del self.sessions[session_id]
        # A request of this session that is still waiting on the backend sees `closed` and skips its commit.
        session.closed = True
        session.manager.clear()

    async def chat(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Serve one OpenAI chat request and return the backend's full reply (meta_info included)."""
        session = self.get_session(session_id)
        request = _resolve_request(request, session)
        messages = request.get("messages") or []
        if not messages:
            raise ProxyError(400, "messages must not be empty")
        tools = request.get("tools") or None
        template_kwargs = {
            **(self.config.apply_chat_template_kwargs or {}),
            **(request.get("chat_template_kwargs") or {}),
        }
        try:
            prepared = session.manager.prepare(
                messages,
                tools=tools,
                builder=self._builder(template_kwargs),
                context_key=context_key(tools, template_kwargs),
            )
        except ValueError as exc:
            # Continuous token refused the request, e.g. an appended role or a template it cannot diff.
            raise ProxyError(400, f"cannot build the prompt incrementally: {exc}") from exc

        r3_row_start = self._r3_row_start(session, reusable_rows=prepared.reusable_r3_rows)
        body = build_request(request, input_ids=prepared.input_ids, r3_row_start=r3_row_start)
        reply = await generate(self.client, self.config.backend_url, body, routing_key=session_id)
        generation = parse_generation(
            reply, prompt_len=len(prepared.input_ids), r3_row_start=r3_row_start, num_layers=self.config.num_layers
        )
        if not session.closed:
            session.manager.commit(prepared, generation)
        return reply

    def collect(self, session_id: str, *, max_seq_len: int | None, agent_metadata: dict | None) -> bytes:
        """The session's samples, encoded with Miles' samples codec."""
        session = self.get_session(session_id)
        metadata: dict[str, Any] = {"tree": session.manager.tree_metadata()}
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if not session.manager.nodes:
            return encode_samples([], metadata, empty_reason="no_records", fields=COMPUTED_FIELDS_V2)
        try:
            samples = session.manager.leaf_samples(max_seq_len=max_seq_len, tokenizer=self.tokenizer)
            if samples:
                # Default hooks: drop leaves superseded by a retry, give each shared completion loss in
                # exactly one sample, and assign the agent's reward.
                samples = self._sample_postprocessor(self._sample_picker(samples, metadata), metadata)
        except (AssertionError, ValueError) as exc:
            raise ProxyError(422, f"sample export failed: {exc}") from exc
        if not samples:
            return encode_samples([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
        return encode_samples(samples, metadata, fields=COMPUTED_FIELDS_V2)

    async def register(self) -> None:
        """Announce this replica to the Router. Both start at the same time, so retry until it answers."""
        while True:
            try:
                response = await self.client.post(
                    f"{self.config.router_url}/replicas",
                    json={"url": self.config.url, "agent_url": self.config.public_url or self.config.url},
                )
                response.raise_for_status()
                logger.info("Registered %s with router %s", self.config.url, self.config.router_url)
                return
            except httpx.HTTPError as exc:
                logger.info("Router %s not reachable yet (%s); retrying", self.config.router_url, exc)
                await asyncio.sleep(1)

    def _builder(self, template_kwargs: dict[str, Any]) -> ContinuousTokenBuilder:
        # Builders hold no per-session state, so one per distinct set of template kwargs is enough.
        key = json.dumps(template_kwargs, sort_keys=True, default=str)
        if key not in self._builders:
            self._builders[key] = create_builder(
                self.tokenizer, model_type=self.model_type, chat_template_kwargs=template_kwargs
            )
        return self._builders[key]

    def _r3_row_start(self, session: Session, *, reusable_rows: int) -> int | None:
        """First routing-replay row to ask the backend for, or None when replay is off."""
        if session.evaluation or not self.config.use_rollout_routing_replay:
            return None
        # Additional mode asks only for the rows after what the parent already holds; full mode asks
        # for every row, because "retract" may have recomputed earlier ones.
        return reusable_rows if self.config.use_addition_r3 else 0


def _resolve_request(request: dict[str, Any], session: Session) -> dict[str, Any]:
    """Fill the sampling fields the client left out from the session's defaults; refuse fields the proxy owns."""
    if request.get("input_ids") is not None:
        raise ProxyError(400, "input_ids are rendered by the proxy and must not be sent")
    if (request.get("n") or 1) != 1:
        raise ProxyError(400, "only n=1 is supported")
    request = dict(request)
    for key, value in session.sampling_defaults.items():
        # A training session always samples at the run's temperature: the trainer treats rollout logprobs as
        # sampled at it, and a black-box agent such as Claude Code cannot be told to change its own setting.
        if request.get(key) is None or (key == "temperature" and not session.evaluation):
            request[key] = value
    return request


async def _json_body(request: Request) -> dict[str, Any]:
    body = await request.body()
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise ProxyError(400, f"invalid JSON body: {exc}") from exc


def _openai_response(reply: dict[str, Any], *, stream: bool, logprobs: bool) -> Response:
    """The reply as the agent sees it: without meta_info, and without logprobs unless it asked.

    ``stream`` is served as one SSE chunk plus [DONE] once generation is complete: the backend
    call itself cannot stream, because the turn is committed from the complete reply.
    """
    body = _client_response_body(reply, ClientResponseIntent(stream=stream, logprobs=logprobs))
    if stream:
        sse = b"data: " + _render_json(_response_to_stream_chunk(body)) + b"\n\ndata: [DONE]\n\n"
        return Response(
            content=sse,
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
    return Response(content=_render_json(body), media_type=JSON_MEDIA_TYPE)


def create_app(replica: Replica) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        registration = asyncio.create_task(replica.register()) if replica.config.router_url else None
        yield
        if registration is not None:
            registration.cancel()
        await replica.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(_request: Request, exc: ProxyError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    @app.get("/health")
    async def health():
        return {"status": "ok", "num_sessions": len(replica.sessions)}

    @app.post("/sessions")
    async def create_session(request: Request):
        try:
            params = CreateSessionRequest.model_validate_json(await request.body() or b"{}")
        except ValidationError as exc:
            raise ProxyError(400, str(exc)) from exc
        return {"session_id": replica.create_session(params)}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        replica.delete_session(session_id)
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        body = await _json_body(request)
        reply = await replica.chat(session_id, body)
        return _openai_response(reply, stream=bool(body.get("stream")), logprobs=bool(body.get("logprobs")))

    add_anthropic_route(app, replica)

    @app.post("/sessions/{session_id}/samples")
    async def collect_samples(session_id: str, request: Request):
        body = await _json_body(request)
        payload = replica.collect(session_id, max_seq_len=body.get("max_seq_len"), agent_metadata=body.get("metadata"))
        return Response(content=payload, media_type="application/octet-stream")

    # Registered last so the routes above match first.
    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST"])
    async def passthrough(session_id: str, path: str, request: Request):
        """Anything else an agent calls (e.g. Anthropic count_tokens) goes to the backend as is and is not recorded."""
        replica.get_session(session_id)
        response = await replica.client.request(
            request.method,
            f"{replica.config.backend_url}/{path}",
            content=await request.body(),
            headers={"content-type": request.headers.get("content-type", JSON_MEDIA_TYPE)},
        )
        return Response(
            content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type")
        )

    return app


def run_replica(config: ReplicaConfig) -> None:
    # Spawned as a fresh interpreter, so logging and the process title are set up here.
    configure_logger_raw("agent_proxy_replica")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    setproctitle.setproctitle("miles-agent-proxy-replica")

    tokenizer = load_tokenizer(
        config.hf_checkpoint, chat_template_path=config.chat_template_path, trust_remote_code=True
    )
    # The root config.json model_type selects the continuous token builder.
    model_type = AutoConfig.from_pretrained(config.hf_checkpoint, trust_remote_code=True).model_type
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=1024), timeout=httpx.Timeout(config.timeout))
    replica = Replica(config, tokenizer=tokenizer, model_type=model_type, client=client)
    logger.info("Proxy replica on %s, backend %s, model_type=%s", config.url, config.backend_url, model_type)
    uvicorn.run(
        create_app(replica),
        # Listen on every interface: a tunnel (public_url) may reach the port through loopback, while the
        # Router and the Miles driver use host:port.
        host="0.0.0.0",
        port=config.port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=60,
    )


def main(argv: list[str] | None = None) -> None:
    run_replica(parse_config_argv(ReplicaConfig, argv))


if __name__ == "__main__":
    main()
