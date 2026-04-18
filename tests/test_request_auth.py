from __future__ import annotations

import datetime as _dt
import time

import pytest

from eirel.request_auth import (
    CLOCK_SKEW_SECONDS,
    InboundAuthError,
    _NonceCache,
    set_keypair_verifier,
    verify_inbound_request,
)
from eirel.signing import build_signing_string, sha256_hex


def _iso_now(offset_seconds: float = 0.0) -> str:
    return (_dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=offset_seconds)).isoformat()


def _valid_headers(**overrides) -> dict[str, str]:
    base = {
        "X-Hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "X-Signature": "0x" + "aa" * 64,
        "X-Timestamp": _iso_now(),
        "X-Request-Id": f"req-{time.time_ns()}",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _accept_all_signatures(monkeypatch):
    # Bittensor is an optional dep; swap in a deterministic test verifier.
    set_keypair_verifier(lambda hotkey, message, signature: True)
    monkeypatch.delenv("EIREL_DISABLE_REQUEST_AUTH", raising=False)
    monkeypatch.delenv("EIREL_ALLOWED_VALIDATOR_HOTKEYS", raising=False)
    yield
    set_keypair_verifier(None)


def test_verify_inbound_request_accepts_valid_signed_request():
    cache = _NonceCache()
    hotkey = verify_inbound_request(
        method="POST",
        path="/v1/agent/infer",
        headers=_valid_headers(),
        body=b'{"task_id":"t1"}',
        nonce_cache=cache,
    )
    assert hotkey.startswith("5FHneW")


def test_verify_inbound_request_rejects_missing_headers():
    with pytest.raises(InboundAuthError, match="missing auth header"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers={"X-Hotkey": "abc"},
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verify_inbound_request_rejects_stale_timestamp():
    stale_headers = _valid_headers()
    stale_headers["X-Timestamp"] = _iso_now(offset_seconds=-(CLOCK_SKEW_SECONDS + 10))
    with pytest.raises(InboundAuthError, match="outside"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=stale_headers,
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verify_inbound_request_rejects_future_timestamp():
    headers = _valid_headers()
    headers["X-Timestamp"] = _iso_now(offset_seconds=CLOCK_SKEW_SECONDS + 10)
    with pytest.raises(InboundAuthError, match="outside"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=headers,
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verify_inbound_request_rejects_replayed_request_id():
    cache = _NonceCache()
    headers = _valid_headers()
    verify_inbound_request(
        method="POST",
        path="/v1/agent/infer",
        headers=headers,
        body=b"{}",
        nonce_cache=cache,
    )
    with pytest.raises(InboundAuthError, match="replay"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=headers,
            body=b"{}",
            nonce_cache=cache,
        )


def test_verify_inbound_request_rejects_invalid_signature_format():
    headers = _valid_headers()
    headers["X-Signature"] = "not-hex"
    with pytest.raises(InboundAuthError, match="hex string"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=headers,
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verify_inbound_request_rejects_when_verifier_returns_false():
    set_keypair_verifier(lambda hotkey, message, signature: False)
    with pytest.raises(InboundAuthError, match="signature verification failed"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=_valid_headers(),
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verify_inbound_request_bypasses_when_env_flag_set(monkeypatch):
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")
    hotkey = verify_inbound_request(
        method="POST",
        path="/v1/agent/infer",
        headers={"X-Hotkey": "anyone"},
        body=b"{}",
        nonce_cache=_NonceCache(),
    )
    assert hotkey == "anyone"


def test_verify_inbound_request_respects_allowlist(monkeypatch):
    monkeypatch.setenv(
        "EIREL_ALLOWED_VALIDATOR_HOTKEYS",
        "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    )
    # Allowlisted hotkey is accepted.
    verify_inbound_request(
        method="POST",
        path="/v1/agent/infer",
        headers=_valid_headers(),
        body=b"{}",
        nonce_cache=_NonceCache(),
    )
    # Non-allowlisted hotkey is rejected.
    with pytest.raises(InboundAuthError, match="allowlist"):
        verify_inbound_request(
            method="POST",
            path="/v1/agent/infer",
            headers=_valid_headers(**{"X-Hotkey": "5Grwv..."}),
            body=b"{}",
            nonce_cache=_NonceCache(),
        )


def test_verifier_receives_canonical_signing_string():
    captured: dict[str, str] = {}

    def _capture(hotkey: str, message: str, signature: bytes) -> bool:
        captured["hotkey"] = hotkey
        captured["message"] = message
        captured["signature_len"] = str(len(signature))
        return True

    set_keypair_verifier(_capture)
    headers = _valid_headers()
    body = b'{"hello":"world"}'
    verify_inbound_request(
        method="POST",
        path="/v1/agent/infer",
        headers=headers,
        body=body,
        nonce_cache=_NonceCache(),
    )
    expected = build_signing_string(
        "POST", "/v1/agent/infer", sha256_hex(body), headers["X-Timestamp"]
    )
    assert captured["message"] == expected
    assert int(captured["signature_len"]) == 64


def test_nonce_cache_evicts_oldest():
    cache = _NonceCache(max_size=3)
    cache.check_and_add("a")
    cache.check_and_add("b")
    cache.check_and_add("c")
    cache.check_and_add("d")
    # "a" should have been evicted, so re-adding it is allowed.
    cache.check_and_add("a")
