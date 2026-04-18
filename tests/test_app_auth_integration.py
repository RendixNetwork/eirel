from __future__ import annotations

import datetime as _dt

from fastapi.testclient import TestClient

from eirel import AgentCapabilityMetadata, AgentInvocationRequest, AgentInvocationResponse, BaseAgent
from eirel.app import MinerApp
from eirel.agent_server import build_agent_app
from eirel.request_auth import _default_nonce_cache, set_keypair_verifier
from eirel.signing import build_signing_string, sha256_hex


def _iso_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _signed_headers(method: str, path: str, body: bytes, *, request_id: str) -> dict[str, str]:
    timestamp = _iso_now()
    return {
        "X-Hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "X-Signature": "0x" + "bb" * 64,
        "X-Timestamp": timestamp,
        "X-Request-Id": request_id,
        "Content-Type": "application/json",
    }


async def _handler(payload: dict) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}


def _setup(monkeypatch) -> None:
    monkeypatch.delenv("EIREL_DISABLE_REQUEST_AUTH", raising=False)
    monkeypatch.delenv("EIREL_ALLOWED_VALIDATOR_HOTKEYS", raising=False)
    _default_nonce_cache.clear()
    set_keypair_verifier(lambda hk, msg, sig: True)


def test_miner_app_rejects_unsigned_chat_request(monkeypatch):
    _setup(monkeypatch)
    app = MinerApp(title="test", handler=_handler).fastapi_app()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401
    assert "missing auth header" in response.json()["detail"].lower()
    set_keypair_verifier(None)


def test_miner_app_accepts_signed_chat_request(monkeypatch):
    _setup(monkeypatch)
    app = MinerApp(title="test", handler=_handler).fastapi_app()
    client = TestClient(app)
    body = b'{"messages":[{"role":"user","content":"hi"}]}'
    headers = _signed_headers("POST", "/v1/chat/completions", body, request_id="signed-1")
    response = client.post("/v1/chat/completions", content=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    set_keypair_verifier(None)


def test_miner_app_healthz_remains_open(monkeypatch):
    _setup(monkeypatch)
    app = MinerApp(title="test", handler=_handler).fastapi_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    set_keypair_verifier(None)


def test_miner_app_deep_healthz_reports_provider_state(monkeypatch):
    _setup(monkeypatch)
    from eirel.provider import MinerProviderConfig

    config = MinerProviderConfig(
        provider="openai", model="gpt-4o-mini", mode="direct", api_key="k"
    )
    app = MinerApp(title="test", handler=_handler, provider_config=config).fastapi_app()
    client = TestClient(app)
    response = client.get("/healthz?deep=1")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "provider": "configured"}

    # Now break the config → deep check should report degraded.
    config.api_key = None
    response = client.get("/healthz?deep=1")
    body = response.json()
    assert body["status"] == "degraded"
    assert "misconfigured" in body["provider"]
    set_keypair_verifier(None)


def test_agent_server_rejects_unsigned_infer(monkeypatch):
    _setup(monkeypatch)

    class Dummy(BaseAgent):
        async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
            return AgentInvocationResponse(task_id=request.task_id, family_id=request.family_id)

    agent = Dummy(
        hotkey="hk",
        endpoint="http://127.0.0.1:9000",
        version="1.0.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat", description="d", latency_ms_p50=1000
        ),
    )
    client = TestClient(build_agent_app(agent))
    response = client.post("/v1/agent/infer", json={"task_id": "t", "primary_goal": "g", "subtask": "s", "family_id": "general_chat"})
    assert response.status_code == 401
    set_keypair_verifier(None)


def test_disable_auth_env_bypass(monkeypatch):
    _setup(monkeypatch)
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")
    app = MinerApp(title="test", handler=_handler).fastapi_app()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    set_keypair_verifier(None)
