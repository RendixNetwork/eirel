from __future__ import annotations

"""Tests for Item 2: Resume token HMAC signing and verification."""

import time

import pytest

from eirel.token_signing import InvalidResumeToken, sign_resume_token, verify_resume_token


SECRET = "test-validator-secret-key-2026"


# ── Round-trip ───────────────────────────────────────────────────────────────


def test_sign_and_verify_round_trip():
    payload = "resume-abc-123"
    signed = sign_resume_token(payload, SECRET)
    assert verify_resume_token(signed, SECRET) == payload


def test_round_trip_with_unicode_payload():
    payload = "task-\u00e9\u00e8\u00ea-resume"
    signed = sign_resume_token(payload, SECRET)
    assert verify_resume_token(signed, SECRET) == payload


def test_round_trip_with_long_payload():
    payload = "x" * 5000
    signed = sign_resume_token(payload, SECRET)
    assert verify_resume_token(signed, SECRET) == payload


# ── Tampered payload detection ───────────────────────────────────────────────


def test_tampered_payload_detected():
    signed = sign_resume_token("original-payload", SECRET)
    parts = signed.split(":")
    # Replace the first part (base64 payload) with a different value.
    import base64
    fake = base64.urlsafe_b64encode(b"tampered-payload").decode()
    tampered = f"{fake}:{parts[1]}:{parts[2]}"
    with pytest.raises(InvalidResumeToken, match="HMAC verification failed"):
        verify_resume_token(tampered, SECRET)


def test_tampered_mac_detected():
    signed = sign_resume_token("my-payload", SECRET)
    parts = signed.split(":")
    tampered = f"{parts[0]}:{parts[1]}:{'0' * 64}"
    with pytest.raises(InvalidResumeToken, match="HMAC verification failed"):
        verify_resume_token(tampered, SECRET)


def test_tampered_timestamp_detected():
    signed = sign_resume_token("my-payload", SECRET)
    parts = signed.split(":")
    tampered = f"{parts[0]}:9999999999.000000:{parts[2]}"
    with pytest.raises(InvalidResumeToken, match="HMAC verification failed"):
        verify_resume_token(tampered, SECRET)


def test_wrong_secret_rejected():
    signed = sign_resume_token("my-payload", SECRET)
    with pytest.raises(InvalidResumeToken, match="HMAC verification failed"):
        verify_resume_token(signed, "wrong-secret")


# ── Expired token rejection ──────────────────────────────────────────────────


def test_expired_token_rejected():
    old_time = time.time() - 7200  # 2 hours ago
    signed = sign_resume_token("my-payload", SECRET, issued_at=old_time)
    with pytest.raises(InvalidResumeToken, match="token expired"):
        verify_resume_token(signed, SECRET, max_age_seconds=3600)


def test_fresh_token_accepted():
    signed = sign_resume_token("my-payload", SECRET, issued_at=time.time())
    result = verify_resume_token(signed, SECRET, max_age_seconds=3600)
    assert result == "my-payload"


def test_future_token_rejected():
    future_time = time.time() + 600  # 10 minutes in the future
    signed = sign_resume_token("my-payload", SECRET, issued_at=future_time)
    with pytest.raises(InvalidResumeToken, match="future"):
        verify_resume_token(signed, SECRET)


# ── Malformed tokens ────────────────────────────────────────────────────────


def test_malformed_token_no_colons():
    with pytest.raises(InvalidResumeToken, match="malformed"):
        verify_resume_token("no-colons-here", SECRET)


def test_malformed_token_too_few_parts():
    with pytest.raises(InvalidResumeToken, match="malformed"):
        verify_resume_token("part1:part2", SECRET)


def test_malformed_token_too_many_parts():
    with pytest.raises(InvalidResumeToken, match="malformed"):
        verify_resume_token("a:b:c:d", SECRET)


# ── None / empty passthrough ────────────────────────────────────────────────


def test_empty_secret_raises_on_sign():
    with pytest.raises(ValueError, match="secret must not be empty"):
        sign_resume_token("payload", "")


def test_empty_secret_raises_on_verify():
    with pytest.raises(ValueError, match="secret must not be empty"):
        verify_resume_token("a:b:c", "")


# ── Token format ─────────────────────────────────────────────────────────────


def test_token_has_three_colon_separated_parts():
    signed = sign_resume_token("test", SECRET)
    parts = signed.split(":")
    assert len(parts) == 3


def test_token_mac_is_64_hex_chars():
    signed = sign_resume_token("test", SECRET)
    mac = signed.split(":")[2]
    assert len(mac) == 64
    assert all(c in "0123456789abcdef" for c in mac)


# ── helpers.py integration ───────────────────────────────────────────────────


def test_helpers_maybe_sign_token():
    from eirel.helpers import _maybe_sign_token

    # No secret → passthrough
    assert _maybe_sign_token("raw-token", None) == "raw-token"
    assert _maybe_sign_token(None, SECRET) is None

    # With secret → signed
    signed = _maybe_sign_token("raw-token", SECRET)
    assert signed is not None
    assert signed != "raw-token"
    assert verify_resume_token(signed, SECRET) == "raw-token"


def test_workflow_deferred_response_signs_token():
    from eirel.helpers import workflow_deferred_response

    resp = workflow_deferred_response(
        task_id="t1",
        family_id="general_chat",
        output={"summary": "test"},
        resume_token="my-token",
        checkpoint_events=[],
        runtime_state_patch={},
        signing_secret=SECRET,
    )
    # Token should be signed (not the raw value).
    assert resp.resume_token != "my-token"
    assert verify_resume_token(resp.resume_token, SECRET) == "my-token"


def test_workflow_completed_response_signs_token():
    from eirel.helpers import workflow_completed_response

    resp = workflow_completed_response(
        task_id="t1",
        family_id="general_chat",
        output={"summary": "test"},
        resume_token="my-token",
        signing_secret=SECRET,
    )
    assert resp.resume_token != "my-token"
    assert verify_resume_token(resp.resume_token, SECRET) == "my-token"


def test_workflow_completed_response_no_secret_passthrough():
    from eirel.helpers import workflow_completed_response

    resp = workflow_completed_response(
        task_id="t1",
        family_id="general_chat",
        output={"summary": "test"},
        resume_token="my-token",
    )
    assert resp.resume_token == "my-token"
