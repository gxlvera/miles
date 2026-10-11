"""Router + two replicas + a fake SGLang backend, all in one process over ASGI.

Checks registration, that a session lives on exactly one replica, prefix reuse across turns,
the client reply shape, collect through Miles' own samples decoder, and delete.
Needs SGLang (the replica imports Miles modules that depend on it) and the Qwen3.5 tokenizer.
"""

import asyncio

import httpx
import pytest

pytest.importorskip("sglang")

from agent_proxy import router  # noqa: E402
from agent_proxy.server import Replica, ReplicaConfig, create_app  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample  # noqa: E402
from miles.utils.types import Sample  # noqa: E402


class _Hosts(httpx.AsyncBaseTransport):
    """Routes each request to the ASGI app registered for its host name."""

    def __init__(self):
        self.apps: dict[str, FastAPI] = {}

    async def handle_async_request(self, request):
        return await httpx.ASGITransport(app=self.apps[request.url.host]).handle_async_request(request)


def _fake_backend(tokenizer, requests: list[dict]) -> FastAPI:
    app = FastAPI()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        requests.append(body)
        text = f"reply {len(requests)}"
        output = tokenizer.encode(text, add_special_tokens=False) + [im_end]
        return {
            "id": f"chatcmpl-{len(requests)}",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                    "meta_info": {
                        "output_token_logprobs": [[-0.5, token, None] for token in output],
                        "completion_tokens": len(output),
                        "weight_version": 1,
                    },
                }
            ],
        }

    return app


def test_router_replicas_and_session_lifecycle(tokenizer):
    asyncio.run(_scenario(tokenizer))


async def _scenario(tokenizer):
    hosts = _Hosts()
    client = httpx.AsyncClient(transport=hosts)
    backend_requests: list[dict] = []
    hosts.apps["backend"] = _fake_backend(tokenizer, backend_requests)
    hosts.apps["router"] = router.create_app(client)
    replicas = []
    for name in ("replica0", "replica1"):
        config = ReplicaConfig(
            host=name, port=80, router_url="http://router", backend_url="http://backend", hf_checkpoint="unused"
        )
        replica = Replica(config, tokenizer=tokenizer, model_type="qwen3_5", client=client)
        hosts.apps[name] = create_app(replica)
        replicas.append(replica)
        await replica.register()
    assert len((await client.get("http://router/replicas")).json()["replicas"]) == 2

    handles = [(await client.post("http://router/sessions", json={"temperature": 1.0})).json() for _ in range(6)]
    for handle in handles:
        owners = [replica for replica in replicas if handle["session_id"] in replica.sessions]
        assert [owner.config.url for owner in owners] == [handle["replica_url"]]

    handle = handles[0]
    chat_url = f"{handle['agent_base_url']}/v1/chat/completions"
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."}]
    first = (await client.post(chat_url, json={"model": "m", "messages": messages})).json()
    assert "meta_info" not in first["choices"][0] and "logprobs" not in first["choices"][0]
    assert backend_requests[0]["temperature"] == 1.0  # filled from the session defaults

    messages += [first["choices"][0]["message"], {"role": "user", "content": "Again."}]
    await client.post(chat_url, json={"model": "m", "messages": messages})
    first_prompt, second_prompt = backend_requests[0]["input_ids"], backend_requests[1]["input_ids"]
    first_tokens = first_prompt + tokenizer.encode("reply 1", add_special_tokens=False)
    assert second_prompt[: len(first_tokens) + 1] == first_tokens + [tokenizer.convert_tokens_to_ids("<|im_end|>")]

    other = next(url for url in ("http://replica0:80", "http://replica1:80") if url != handle["replica_url"])
    wrong_owner = await client.post(f"{other}/sessions/{handle['session_id']}/v1/chat/completions", json={})
    assert wrong_owner.status_code == 404

    session_url = f"{handle['replica_url']}/sessions/{handle['session_id']}"
    payload = (await client.post(f"{session_url}/samples", json={"metadata": {"reward": 1.0}})).content
    reply = decode_samples_and_merge_input_sample(payload, Sample(index=0, prompt="p"), fields=COMPUTED_FIELDS_V2)
    (sample,) = reply.samples
    assert sample.tokens == second_prompt + tokenizer.encode("reply 2", add_special_tokens=False) + [
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    assert sample.reward == 1.0
    assert sum(sample.loss_mask) == 2 * (len(tokenizer.encode("reply 1", add_special_tokens=False)) + 1)

    assert (await client.delete(session_url)).status_code == 204
    assert (await client.post(chat_url, json={"model": "m", "messages": messages})).status_code == 404
