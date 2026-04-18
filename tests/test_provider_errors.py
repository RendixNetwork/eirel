from __future__ import annotations

from typing import Any

import httpx
import pytest

from eirel.provider import (
    AgentProviderClient,
    MinerProviderConfig,
    ProviderQuotaExceeded,
    ProviderRequestError,
)


def _config(**overrides) -> MinerProviderConfig:
    defaults = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "mode": "direct",
        "api_key": "test-key",
    }
    return MinerProviderConfig(**{**defaults, **overrides})


# ── fail-fast on missing API key ────────────────────────────────────────────


def test_direct_request_parts_raises_when_api_key_missing():
    client = AgentProviderClient(_config(api_key=None))
    with pytest.raises(RuntimeError, match="MINER_API_KEY is not set"):
        client._direct_request_parts({"messages": []})


def test_direct_request_parts_embeds_key_when_present():
    client = AgentProviderClient(_config(api_key="sk-real"))
    url, headers, body = client._direct_request_parts({"messages": []})
    assert "sk-real" in headers["Authorization"]
    assert body["model"] == "gpt-4o-mini"


# ── sanitized HTTP errors ───────────────────────────────────────────────────


def _fake_response(status_code: int, text: str) -> httpx.Response:
    return httpx.Response(status_code=status_code, text=text, request=httpx.Request("POST", "https://x"))


def test_raise_for_status_wraps_in_provider_request_error():
    client = AgentProviderClient(_config())
    with pytest.raises(ProviderRequestError) as excinfo:
        client._raise_for_status(_fake_response(500, "upstream died"))
    assert "500" in str(excinfo.value)
    assert "upstream died" in str(excinfo.value)
    # `from None` means __cause__ is None → httpx exception chain not leaked.
    assert excinfo.value.__cause__ is None


def test_raise_for_status_truncates_long_body():
    client = AgentProviderClient(_config())
    huge = "x" * 5000
    with pytest.raises(ProviderRequestError) as excinfo:
        client._raise_for_status(_fake_response(502, huge))
    assert len(str(excinfo.value)) < 1000  # sanitized excerpt + preamble


def test_raise_for_status_passthrough_on_success():
    client = AgentProviderClient(_config())
    # Should not raise.
    client._raise_for_status(_fake_response(200, "ok"))


# ── validate_for_runtime ────────────────────────────────────────────────────


def test_validate_for_runtime_direct_mode_requires_api_key():
    with pytest.raises(RuntimeError, match="MINER_API_KEY"):
        _config(api_key=None).validate_for_runtime()


def test_validate_for_runtime_proxy_mode_requires_proxy_creds():
    with pytest.raises(RuntimeError, match="PROXY"):
        _config(mode="proxy").validate_for_runtime()


def test_validate_for_runtime_auto_mode_requires_either():
    with pytest.raises(RuntimeError, match="auto"):
        _config(mode="auto", api_key=None).validate_for_runtime()


def test_validate_for_runtime_auto_mode_passes_with_api_key():
    _config(mode="auto").validate_for_runtime()  # no raise


def test_validate_for_runtime_auto_mode_passes_with_proxy():
    _config(
        mode="auto",
        api_key=None,
        subnet_proxy_url="https://proxy.example.com",
        subnet_proxy_token="t",
    ).validate_for_runtime()  # no raise


# ── quota tracking ──────────────────────────────────────────────────────────


async def test_quota_blocks_further_requests_after_max_requests():
    client = AgentProviderClient(_config(max_requests=2))
    # Manually increment counter to simulate past calls.
    client._request_count = 2
    with pytest.raises(ProviderQuotaExceeded, match="request quota"):
        await client.chat_completions({"messages": []})


async def test_quota_blocks_further_requests_after_max_tokens():
    client = AgentProviderClient(_config(max_total_tokens=1000))
    client._token_count = 1500
    with pytest.raises(ProviderQuotaExceeded, match="token quota"):
        await client.chat_completions({"messages": []})


def test_quota_reset():
    client = AgentProviderClient(_config())
    client._request_count = 5
    client._token_count = 500
    client.reset_quota()
    assert client.request_count == 0
    assert client.token_count == 0


def test_extract_usage_tokens_openai_shape():
    assert (
        AgentProviderClient._extract_usage_tokens({"usage": {"total_tokens": 42}}) == 42
    )


def test_extract_usage_tokens_anthropic_shape():
    assert (
        AgentProviderClient._extract_usage_tokens(
            {"usage": {"input_tokens": 10, "output_tokens": 20}}
        )
        == 30
    )


def test_extract_usage_tokens_missing_is_zero():
    assert AgentProviderClient._extract_usage_tokens({}) == 0
    assert AgentProviderClient._extract_usage_tokens({"usage": None}) == 0


# ── run budget header forwarding ────────────────────────────────────────────


def test_from_env_parses_run_budget_usd(monkeypatch):
    monkeypatch.setenv("EIREL_RUN_BUDGET_USD", "12.5")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "https://proxy.example.com")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "t")
    cfg = MinerProviderConfig.from_env()
    assert cfg.run_budget_usd == 12.5


def test_from_env_run_budget_missing_is_none(monkeypatch):
    monkeypatch.delenv("EIREL_RUN_BUDGET_USD", raising=False)
    monkeypatch.setenv("MINER_API_KEY", "k")
    cfg = MinerProviderConfig.from_env()
    assert cfg.run_budget_usd is None


def test_from_env_run_budget_invalid_is_none(monkeypatch):
    monkeypatch.setenv("EIREL_RUN_BUDGET_USD", "not-a-number")
    monkeypatch.setenv("MINER_API_KEY", "k")
    cfg = MinerProviderConfig.from_env()
    assert cfg.run_budget_usd is None


def test_from_env_reads_all_quota_knobs(monkeypatch):
    monkeypatch.setenv("MINER_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_MAX_REQUESTS", "500")
    monkeypatch.setenv("EIREL_PROVIDER_MAX_TOTAL_TOKENS", "1500000")
    monkeypatch.setenv("EIREL_PROVIDER_MAX_WALL_CLOCK_SECONDS", "1800")
    monkeypatch.setenv("EIREL_PROVIDER_PER_REQUEST_TIMEOUT_SECONDS", "45")
    cfg = MinerProviderConfig.from_env()
    assert cfg.max_requests == 500
    assert cfg.max_total_tokens == 1500000
    assert cfg.max_wall_clock_seconds == 1800
    assert cfg.per_request_timeout_seconds == 45


def test_from_env_quota_defaults_when_missing(monkeypatch):
    monkeypatch.setenv("MINER_API_KEY", "k")
    for key in (
        "EIREL_PROVIDER_MAX_REQUESTS",
        "EIREL_PROVIDER_MAX_TOTAL_TOKENS",
        "EIREL_PROVIDER_MAX_WALL_CLOCK_SECONDS",
        "EIREL_PROVIDER_PER_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = MinerProviderConfig.from_env()
    # Defaults are deliberately conservative; production pods override them
    # via env in infra/miner_runtime/runtime_manager.py.
    assert cfg.max_requests == 24
    assert cfg.max_total_tokens == 60_000


def test_from_env_quota_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MINER_API_KEY", "k")
    monkeypatch.setenv("EIREL_PROVIDER_MAX_REQUESTS", "not-a-number")
    cfg = MinerProviderConfig.from_env()
    assert cfg.max_requests == 24


async def test_proxy_chat_completions_forwards_run_budget_header(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["headers"] = headers
            return httpx.Response(
                status_code=200,
                json={"upstream_response": {"choices": []}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = AgentProviderClient(
        _config(
            mode="proxy",
            api_key=None,
            subnet_proxy_url="https://proxy.example.com",
            subnet_proxy_token="tok",
            subnet_proxy_job_id="job-xyz",
            run_budget_usd=17.25,
        )
    )
    await client._proxy_chat_completions({"messages": []})
    assert captured["headers"]["X-Eirel-Job-Id"] == "job-xyz"
    assert captured["headers"]["X-Eirel-Run-Budget-Usd"] == "17.250000"


async def test_proxy_chat_completions_omits_budget_header_when_unset(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json, headers):
            captured["headers"] = headers
            return httpx.Response(
                status_code=200,
                json={"upstream_response": {"choices": []}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = AgentProviderClient(
        _config(
            mode="proxy",
            api_key=None,
            subnet_proxy_url="https://proxy.example.com",
            subnet_proxy_token="tok",
            run_budget_usd=None,
        )
    )
    await client._proxy_chat_completions({"messages": []})
    assert "X-Eirel-Run-Budget-Usd" not in captured["headers"]
