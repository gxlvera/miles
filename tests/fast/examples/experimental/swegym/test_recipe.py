"""CPU checks for evaluation isolation, credential routing and reward integrity."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

RECIPE = Path(__file__).resolve().parents[5] / "examples/experimental/swegym"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    result = {}
    for name in ["data_utils", "sandbox", "agent", "gateway"]:
        spec = importlib.util.spec_from_file_location(name, RECIPE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[name] = module
    return SimpleNamespace(**result)


def test_template_cache_invalidation(modules):
    task = {"image": "repo/image@sha256:abc", "sanitize": "reset base", "instance_id": "owner__repo-1"}
    first = modules.sandbox.template_alias(task)
    assert modules.sandbox.template_alias(dict(task)) == first
    assert modules.sandbox.template_alias({**task, "sanitize": "reset other"}) != first


def test_verifier_rejects_missing_or_vacuous_results(modules):
    with pytest.raises(RuntimeError):
        modules.sandbox.parse_verdict("test runner crashed")
    with pytest.raises(RuntimeError):
        modules.sandbox.parse_verdict("MILES_SWEGYM_REWARD_JSON=" + json.dumps({"resolved": True, "parsed_tests": 0, "fail_to_pass_total": 1}))


async def test_fresh_verifier_is_destroyed_even_on_failure(modules, monkeypatch):
    sb = SimpleNamespace(commands=SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("network"))), kill=AsyncMock())
    create = AsyncMock(return_value=sb)
    monkeypatch.setattr(modules.sandbox, "create_sandbox", create)
    with pytest.raises(RuntimeError, match="network"):
        await modules.sandbox.evaluate({"instance_id": "one", "evaluate": "test"}, "template", "")
    create.assert_awaited_once_with("template", task_id="one", role="verifier")
    sb.kill.assert_awaited_once()


def test_modal_url_preserves_session_and_uses_tls(modules):
    assert modules.agent.public_proxy_url("http://10.0.0.1:30000/sessions/abc", "https://example.modal.host") == ("https://example.modal.host/sessions/abc")
    with pytest.raises(ValueError):
        modules.agent.public_proxy_url("http://10.0.0.1:30000/sessions/abc", "http://example.modal.host")


def test_test_edits_cannot_be_rewarded(modules):
    assert modules.agent.forbidden_changes(["pydantic/main.py", "tests/test_main.py", "pyproject.toml"]) == [
        "tests/test_main.py",
        "pyproject.toml",
    ]


async def test_infrastructure_failures_are_not_negative_training_samples(modules):
    with pytest.raises(RuntimeError, match="no verified evaluation"):
        await modules.agent.reward(None, SimpleNamespace(metadata={}))


def test_gateway_does_not_expose_session_admin(modules, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    token = "a" * 48
    key = tmp_path / "token"
    key.write_text(token)
    monkeypatch.setenv("SWEGYM_GATEWAY_TOKEN_FILE", str(key))
    with TestClient(modules.gateway.create_app()) as client:
        assert client.post("/sessions/abc/v1/messages", json={}).status_code == 401
        assert client.post("/sessions", headers={"x-api-key": token}).status_code == 404
        assert client.post("/sessions/abc/samples", headers={"x-api-key": token}).status_code == 404


def test_no_key_in_template_recipe(modules):
    commands = "\n".join(modules.sandbox.build_commands({"sanitize": "git reset"}))
    assert "E2B_API_KEY" not in commands
    assert "ANTHROPIC_API_KEY" not in commands
    assert "blackbox_server" not in commands


def test_claude_effort_and_tool_errors_preserve_training_content(modules):
    block = {"type": "tool_result", "tool_use_id": "call_1", "is_error": True, "content": "No such file"}
    request = {"output_config": {"effort": "high"}, "messages": [{"role": "user", "content": [block]}]}
    result = modules.gateway.normalize_claude_request(request)
    assert "output_config" not in result
    converted = result["messages"][0]["content"][0]
    assert converted["tool_use_id"] == "call_1"
    assert converted["content"] == "[Tool execution failed]\nNo such file"
    assert "is_error" not in converted
    assert block["is_error"] is True
    unsupported = {"output_config": {"format": {"type": "json_schema"}}}
    assert modules.gateway.normalize_claude_request(unsupported) == unsupported


def test_token_count_uses_engine_without_creating_session_trajectory(modules, tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    token = "b" * 48
    key = tmp_path / "token"
    key.write_text(token)
    monkeypatch.setenv("SWEGYM_GATEWAY_TOKEN_FILE", str(key))
    calls = []

    def transport(request):
        calls.append(str(request.url))
        assert "x-api-key" not in request.headers
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine:8000"}]})
        assert request.url.path == "/v1/messages/count_tokens"
        assert json.loads(request.content)["messages"][0]["content"] == "Count me"
        return httpx.Response(200, json={"input_tokens": 12})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(modules.gateway.httpx, "AsyncClient", lambda **kwargs: upstream)
    with TestClient(modules.gateway.create_app()) as client:
        response = client.post(
            "/sessions/test/v1/messages/count_tokens",
            headers={"x-api-key": token},
            json={"model": "Qwen3-14B", "messages": [{"role": "user", "content": "Count me"}]},
        )
    assert response.json() == {"input_tokens": 12}
    assert calls == ["http://127.0.0.1:31000/workers", "http://engine:8000/v1/messages/count_tokens"]



def _collection_verdict():
    return {
        "resolved": False, "parsed_tests": 0, "fail_to_pass_total": 1,
        "failure_reason": "test_log_parse_failure",
        "test_log_tail": "test session starts\nERROR collecting tests/test_main.py\nSyntaxError\nInterrupted: 1 error during collection",
    }


@pytest.mark.parametrize("tail", ["network failed", "missing pytest", "test session starts"])
def test_zero_results_without_collection_evidence_are_rejected(modules, tail):
    verdict = {**_collection_verdict(), "test_log_tail": tail}
    with pytest.raises(RuntimeError):
        modules.sandbox.parse_verdict("MILES_SWEGYM_REWARD_JSON=" + json.dumps(verdict), allow_collection_failure=True)


@pytest.mark.parametrize("control_ok", [True, False])
async def test_candidate_collection_failure_requires_fresh_healthy_control(modules, monkeypatch, control_ok):
    candidate = _collection_verdict()
    control = {"resolved": False, "parsed_tests": 169, "fail_to_pass_total": 1} if control_ok else candidate
    def output(verdict):
        return SimpleNamespace(stdout="MILES_SWEGYM_REWARD_JSON=" + json.dumps(verdict))
    patched = SimpleNamespace(
        files=SimpleNamespace(write=AsyncMock()),
        commands=SimpleNamespace(run=AsyncMock(side_effect=[SimpleNamespace(), output(candidate)])),
        kill=AsyncMock(),
    )
    clean = SimpleNamespace(commands=SimpleNamespace(run=AsyncMock(return_value=output(control))), kill=AsyncMock())
    create = AsyncMock(side_effect=[patched, clean])
    monkeypatch.setattr(modules.sandbox, "create_sandbox", create)
    if control_ok:
        result = await modules.sandbox.evaluate({"instance_id": "one", "evaluate": "test"}, "template", "patch")
        assert result["resolved"] is False
        assert result["failure_reason"] == "candidate_test_collection_failure"
        assert result["unpatched_control_parsed_tests"] == 169
    else:
        with pytest.raises(RuntimeError, match="Invalid evaluator"):
            await modules.sandbox.evaluate({"instance_id": "one", "evaluate": "test"}, "template", "patch")
    assert create.await_count == 2
    patched.kill.assert_awaited_once()
    clean.kill.assert_awaited_once()
