from __future__ import annotations

"""Inbound request authentication for miner endpoints.

Validators authenticate to a miner by signing the request with their Bittensor
hotkey.  The signing contract is symmetric with :mod:`eirel.signing`, which
miners use to sign *outbound* requests to the owner API.

Headers required on protected endpoints:

* ``X-Hotkey``      — the validator's SS58 hotkey address
* ``X-Signature``   — ``0x``-prefixed hex signature of the canonical string
* ``X-Timestamp``   — ISO-8601 UTC timestamp (same format Signer emits)
* ``X-Request-Id``  — unique id used for replay protection

The canonical string is ``f"{METHOD}{path}{body_sha256_hex}{timestamp}"`` —
identical to :func:`eirel.signing.build_signing_string`.

Environment flags:

* ``EIREL_DISABLE_REQUEST_AUTH=1`` — bypass auth entirely (dev only; logged WARN)
* ``EIREL_ALLOWED_VALIDATOR_HOTKEYS`` — optional comma-separated allowlist
"""

import datetime as _dt
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import HTTPException, Request

from eirel.signing import build_signing_string, sha256_hex

_logger = logging.getLogger(__name__)

CLOCK_SKEW_SECONDS = 60.0
DEFAULT_NONCE_CACHE_SIZE = 10_000
_STARTUP_DISABLE_WARNING_EMITTED = False


class InboundAuthError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=401, detail=detail)


class _NonceCache:
    def __init__(self, max_size: int = DEFAULT_NONCE_CACHE_SIZE) -> None:
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size

    def check_and_add(self, request_id: str) -> None:
        if request_id in self._entries:
            raise InboundAuthError("duplicate X-Request-Id (replay detected)")
        self._entries[request_id] = time.time()
        if len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


_default_nonce_cache = _NonceCache()


KeypairVerifier = Callable[[str, str, bytes], bool]


def _default_keypair_verifier(hotkey: str, message: str, signature: bytes) -> bool:
    try:
        import bittensor as bt
    except ImportError as exc:
        raise InboundAuthError(
            "inbound request auth requires bittensor; install with "
            "'pip install eirel[submit]' or set EIREL_DISABLE_REQUEST_AUTH=1 for local dev"
        ) from exc
    try:
        keypair = bt.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(message, signature))
    except Exception as exc:  # pragma: no cover - bittensor surface varies
        _logger.warning("keypair.verify raised for hotkey %s...: %s", hotkey[:8], exc)
        return False


_verifier: KeypairVerifier = _default_keypair_verifier


def set_keypair_verifier(fn: KeypairVerifier | None) -> None:
    """Swap the keypair verifier (tests only).

    Pass ``None`` to restore the default bittensor-backed verifier.
    """
    global _verifier
    _verifier = fn if fn is not None else _default_keypair_verifier


def _auth_disabled() -> bool:
    value = os.getenv("EIREL_DISABLE_REQUEST_AUTH", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _allowlisted_hotkeys() -> set[str]:
    raw = os.getenv("EIREL_ALLOWED_VALIDATOR_HOTKEYS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _get_header(headers: Mapping[str, str], name: str) -> str:
    # Starlette headers are case-insensitive; plain dicts are not.
    if hasattr(headers, "get") and name in headers:
        return str(headers[name])
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return str(value)
    return ""


def _parse_iso_timestamp(value: str) -> float:
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise InboundAuthError("invalid X-Timestamp format (expected ISO-8601 UTC)") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed.timestamp()


def _decode_signature(signature: str) -> bytes:
    value = signature[2:] if signature.startswith(("0x", "0X")) else signature
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise InboundAuthError("invalid X-Signature (expected hex string)") from exc


def verify_inbound_request(
    *,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    nonce_cache: _NonceCache | None = None,
    clock_skew_seconds: float = CLOCK_SKEW_SECONDS,
) -> str:
    """Verify an inbound signed request.

    Returns the authenticated hotkey on success, raises :class:`InboundAuthError`
    (HTTP 401) otherwise.  When ``EIREL_DISABLE_REQUEST_AUTH`` is set, the function
    returns the value of ``X-Hotkey`` (or empty string) without verification.
    """
    global _STARTUP_DISABLE_WARNING_EMITTED
    if _auth_disabled():
        if not _STARTUP_DISABLE_WARNING_EMITTED:
            _logger.warning(
                "EIREL_DISABLE_REQUEST_AUTH is set — inbound request auth is DISABLED. "
                "Do not use this in production."
            )
            _STARTUP_DISABLE_WARNING_EMITTED = True
        return _get_header(headers, "X-Hotkey")

    hotkey = _get_header(headers, "X-Hotkey")
    signature = _get_header(headers, "X-Signature")
    timestamp = _get_header(headers, "X-Timestamp")
    request_id = _get_header(headers, "X-Request-Id")

    if not hotkey or not signature or not timestamp or not request_id:
        raise InboundAuthError(
            "missing auth header (X-Hotkey, X-Signature, X-Timestamp, X-Request-Id required)"
        )

    allowlist = _allowlisted_hotkeys()
    if allowlist and hotkey not in allowlist:
        _logger.warning("rejected request from non-allowlisted hotkey %s...", hotkey[:8])
        raise InboundAuthError("validator hotkey not in allowlist")

    ts_epoch = _parse_iso_timestamp(timestamp)
    now = time.time()
    if abs(now - ts_epoch) > clock_skew_seconds:
        raise InboundAuthError(
            f"X-Timestamp outside ±{int(clock_skew_seconds)}s window"
        )

    cache = nonce_cache if nonce_cache is not None else _default_nonce_cache
    cache.check_and_add(request_id)

    message = build_signing_string(method, path, sha256_hex(body), timestamp)
    sig_bytes = _decode_signature(signature)
    try:
        verified = _verifier(hotkey, message, sig_bytes)
    except InboundAuthError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("keypair verifier raised: %s", exc)
        raise InboundAuthError("signature verification failed") from None

    if not verified:
        raise InboundAuthError("signature verification failed")

    _logger.debug("accepted request from %s... on %s", hotkey[:8], path)
    return hotkey


async def verify_request_dependency(request: Request) -> dict[str, Any]:
    """FastAPI dependency that verifies the inbound request and returns the parsed JSON body.

    Use as::

        @app.post("/v1/foo")
        async def foo(payload: dict = Depends(verify_request_dependency)): ...
    """
    body = await request.body()
    verify_inbound_request(
        method=request.method,
        path=request.url.path,
        headers=request.headers,
        body=body,
    )
    if not body:
        return {}
    try:
        import json as _json

        return _json.loads(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from None


__all__ = [
    "CLOCK_SKEW_SECONDS",
    "InboundAuthError",
    "KeypairVerifier",
    "set_keypair_verifier",
    "verify_inbound_request",
    "verify_request_dependency",
]
