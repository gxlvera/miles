"""The agent server's Router: keeps the list of registered replicas and places each new session on one of them.

It is only on the session-creation path. The handle it returns names the owning replica, and
the agent's turns, collect and delete all go to that replica directly; the Router never sees
per-turn traffic or trajectories.

    POST /replicas   a replica registers its URL at startup
    GET  /replicas   registered replicas (Miles waits on this before starting rollout)
    POST /sessions   create a session on a random replica, return a SessionHandle
"""

import logging
import random
from dataclasses import asdict

import httpx
import setproctitle
import uvicorn
from agent_server.protocol import SessionHandle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.utils.logging_utils import configure_logger_raw
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


class RouterConfig(FrozenStrictBaseModel):
    host: str
    port: int
    timeout: float | None = None


def create_app(client: httpx.AsyncClient) -> FastAPI:
    app = FastAPI()
    # replica URL -> the URL agents use for it (the same unless the replica sits behind a tunnel)
    agent_urls: dict[str, str] = {}

    @app.get("/health")
    async def health():
        return {"status": "ok", "num_replicas": len(agent_urls)}

    @app.post("/replicas")
    async def register_replica(request: Request):
        body = await request.json()
        url = body["url"]
        if url not in agent_urls:
            logger.info("Replica registered: %s, agents use %s", url, body.get("agent_url", url))
        agent_urls[url] = body.get("agent_url", url)
        return {"num_replicas": len(agent_urls)}

    @app.get("/replicas")
    async def list_replicas():
        return {"replicas": list(agent_urls)}

    @app.post("/sessions")
    async def create_session(request: Request):
        if not agent_urls:
            return JSONResponse(status_code=503, content={"error": "no agent server replica has registered yet"})
        replica_url = random.choice(list(agent_urls))
        # The body (sampling defaults, evaluation) is the replica's business; pass it through untouched.
        response = await client.post(
            f"{replica_url}/sessions", content=await request.body(), headers={"content-type": "application/json"}
        )
        if response.status_code != 200:
            return Response(content=response.content, status_code=response.status_code, media_type="application/json")
        session_id = response.json()["session_id"]
        handle = SessionHandle(
            session_id=session_id,
            replica_url=replica_url,
            agent_base_url=f"{agent_urls[replica_url]}/sessions/{session_id}",
        )
        return asdict(handle)

    return app


def run_router(config: RouterConfig) -> None:
    configure_logger_raw("agent_server_router")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    setproctitle.setproctitle("miles-agent-server-router")
    client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout))
    logger.info("Agent server router on %s:%s", config.host, config.port)
    uvicorn.run(create_app(client), host=config.host, port=config.port, log_level="info", access_log=False)


def main(argv: list[str] | None = None) -> None:
    run_router(parse_config_argv(RouterConfig, argv))


if __name__ == "__main__":
    main()
