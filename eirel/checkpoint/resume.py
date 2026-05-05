"""Bridge between graph checkpoints and the existing resume-token HMAC.

Resume tokens already exist in the SDK as a 48 h HMAC-signed envelope
(``eirel.token_signing``). This module wraps a ``(thread_id,
checkpoint_id)`` pair into that same scheme so a graph run can pause,
emit a resume token to the validator, and continue on the next turn —
without changing the wire schema or introducing a second key material.

A token here is just the existing signed token; the payload is a small
JSON object the SDK can decode and look up in any
:class:`~eirel.checkpoint.base.Checkpointer`.
"""
from __future__ import annotations

import json

from eirel.token_signing import (
    InvalidResumeToken,
    RESUME_TOKEN_TTL_SECONDS,
    sign_resume_token,
    verify_resume_token,
)

__all__ = [
    "InvalidResumeToken",
    "RESUME_TOKEN_TTL_SECONDS",
    "encode_thread_token",
    "decode_thread_token",
]


def encode_thread_token(
    *,
    thread_id: str,
    checkpoint_id: str,
    secret: str,
    extra: dict | None = None,
) -> str:
    """Sign a ``(thread_id, checkpoint_id)`` pair into a resume token.

    ``extra`` lets the caller stash a small bag of additional metadata
    (e.g. validator hotkey, conversation id) that the resume-side will
    receive verbatim. Keep this small — the whole token is HMAC-signed
    and base64-encoded inline; large payloads bloat every multi-turn
    request.
    """
    if not thread_id or not checkpoint_id:
        raise ValueError("encode_thread_token requires non-empty thread_id and checkpoint_id")
    payload = {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
    }
    if extra:
        payload["extra"] = dict(extra)
    return sign_resume_token(json.dumps(payload, separators=(",", ":")), secret)


def decode_thread_token(
    signed_token: str,
    secret: str | list[str],
    *,
    max_age_seconds: int = RESUME_TOKEN_TTL_SECONDS,
) -> dict:
    """Verify a token and return the decoded payload dict.

    Raises :class:`~eirel.token_signing.InvalidResumeToken` for tampered,
    expired, or malformed tokens. Returns a dict carrying at minimum
    ``thread_id`` and ``checkpoint_id``.
    """
    payload_str = verify_resume_token(
        signed_token, secret, max_age_seconds=max_age_seconds
    )
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        raise InvalidResumeToken("token payload is not JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidResumeToken("token payload is not an object")
    if "thread_id" not in payload or "checkpoint_id" not in payload:
        raise InvalidResumeToken("token payload missing thread_id or checkpoint_id")
    return payload
