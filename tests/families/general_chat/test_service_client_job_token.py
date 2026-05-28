"""ToolServiceClient prefers the per-request job token owner-api stamps.

The per-request token is bound to this miner's job, so using it (over any
statically configured master token) is what keeps a miner scoped to its
own job when calling tool services.
"""
from __future__ import annotations

import httpx

from eirel import runtime_context as rc
from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)


def _client(handler, **cfg) -> ToolServiceClient:
    return ToolServiceClient(
        ToolServiceConfig(base_url="http://tool.local", **cfg),
        transport=httpx.MockTransport(handler),
    )


async def test_per_request_job_token_overrides_config_token():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["job"] = request.headers.get("x-eirel-job-id")
        return httpx.Response(200, json={"ok": True})

    client = _client(handler, api_token="master", job_id=None)
    id_tok = rc._set_active_job_id("task-eval=t1;deployment=dep-1")
    tok_tok = rc._set_active_job_token("per-job-token")
    try:
        await client.request(path="/v1/x", payload={})
    finally:
        rc._reset_active_job_token(tok_tok)
        rc._reset_active_job_id(id_tok)

    # Context token wins over the statically configured master token.
    assert captured["auth"] == "Bearer per-job-token"
    assert captured["job"] == "task-eval=t1;deployment=dep-1"


async def test_falls_back_to_config_token_without_context():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    assert rc.get_active_job_token() is None  # no ambient context
    client = _client(handler, api_token="master", job_id="j-1")
    await client.request(path="/v1/x", payload={})
    assert captured["auth"] == "Bearer master"
