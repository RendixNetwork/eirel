from __future__ import annotations

"""HMAC-based signing and verification for resume tokens.

Resume tokens flow from validator → miner → back to validator during
deferred workflow execution.  Without integrity protection a miner can
forge or replay tokens to manipulate multi-turn evaluation state.

Token format:  ``base64(payload):timestamp:hmac_sha256_hex``
"""

import base64
import hashlib
import hmac
import time


class InvalidResumeToken(ValueError):
    """Raised when a resume token fails HMAC verification or is expired."""


def sign_resume_token(
    payload: str,
    secret: str,
    *,
    issued_at: float | None = None,
) -> str:
    """Sign *payload* with HMAC-SHA256 and embed a timestamp.

    Returns ``base64(payload):timestamp:hmac_hex``.
    """
    if not secret:
        raise ValueError("signing secret must not be empty")
    issued_at = issued_at or time.time()
    ts_str = f"{issued_at:.6f}"
    encoded_payload = base64.urlsafe_b64encode(payload.encode()).decode()
    message = f"{encoded_payload}:{ts_str}"
    mac = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"{encoded_payload}:{ts_str}:{mac}"


def verify_resume_token(
    signed_token: str,
    secret: str | list[str],
    *,
    max_age_seconds: int = 172800,
) -> str:
    """Verify a signed resume token and return the original payload string.

    *secret* may be a single string or a list of strings for key rotation.
    When a list is provided each secret is tried in order; the token is
    accepted if any secret produces a valid HMAC.

    Raises :class:`InvalidResumeToken` on tampered, expired, or malformed
    tokens.
    """
    secrets = [secret] if isinstance(secret, str) else list(secret)
    if not secrets or not any(secrets):
        raise ValueError("signing secret must not be empty")
    parts = signed_token.split(":")
    if len(parts) != 3:
        raise InvalidResumeToken("malformed token: expected 3 colon-separated parts")

    encoded_payload, ts_str, provided_mac = parts

    # Try each secret (supports key rotation).
    message = f"{encoded_payload}:{ts_str}"
    matched = False
    for candidate_secret in secrets:
        if not candidate_secret:
            continue
        expected_mac = hmac.new(candidate_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(provided_mac, expected_mac):
            matched = True
            break
    if not matched:
        raise InvalidResumeToken("HMAC verification failed: token is tampered")

    # Check expiry.
    try:
        issued_at = float(ts_str)
    except ValueError as exc:
        raise InvalidResumeToken("malformed timestamp in token") from exc

    age = time.time() - issued_at
    if age > max_age_seconds:
        raise InvalidResumeToken(f"token expired: age {age:.0f}s exceeds max {max_age_seconds}s")
    if age < -60:
        # Allow 60s clock skew, but reject tokens from the far future.
        raise InvalidResumeToken("token timestamp is in the future")

    # Decode payload.
    try:
        payload = base64.urlsafe_b64decode(encoded_payload.encode()).decode()
    except Exception as exc:
        raise InvalidResumeToken("failed to decode payload") from exc

    return payload
