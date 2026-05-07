"""Tests for the checkpoint backends and the resume-token bridge."""
from __future__ import annotations

import json
import os

import pytest

from eirel.checkpoint import (
    MAX_CHECKPOINT_BLOB_BYTES,
    CheckpointBlobTooLarge,
    CheckpointTuple,
    InMemoryCheckpointer,
    InvalidResumeToken,
    decode_thread_token,
    deserialize_state,
    encode_thread_token,
    new_checkpoint_id,
    serialize_state,
)
from eirel.token_signing import RESUME_TOKEN_TTL_SECONDS


# -- Serialization + cap ------------------------------------------------------


def test_serialize_roundtrip():
    state = {"messages": [{"role": "user", "content": "hi"}], "next": "respond"}
    blob = serialize_state(state)
    assert deserialize_state(blob) == state


def test_serialize_rejects_oversized_blob():
    state = {"big": "x" * (MAX_CHECKPOINT_BLOB_BYTES + 1024)}
    with pytest.raises(CheckpointBlobTooLarge) as excinfo:
        serialize_state(state)
    assert excinfo.value.size_bytes > MAX_CHECKPOINT_BLOB_BYTES
    assert excinfo.value.limit_bytes == MAX_CHECKPOINT_BLOB_BYTES


def test_serialize_handles_pydantic_model():
    from pydantic import BaseModel

    class Inner(BaseModel):
        x: int

    blob = serialize_state({"obj": Inner(x=7)})
    assert deserialize_state(blob) == {"obj": {"x": 7}}


def test_serialize_rejects_unsupported_type():
    class Custom:
        pass

    with pytest.raises(TypeError):
        serialize_state({"weird": Custom()})


# -- InMemoryCheckpointer contract -------------------------------------------


async def test_in_memory_aput_aget_roundtrip():
    cp = InMemoryCheckpointer()
    state = {"messages": [{"role": "user", "content": "hi"}]}
    cid = new_checkpoint_id()
    tup = await cp.aput(
        thread_id="t1",
        checkpoint_id=cid,
        parent_id=None,
        node="planner",
        state=state,
        metadata={"step": 1},
    )
    assert isinstance(tup, CheckpointTuple)
    fetched = await cp.aget("t1", cid)
    assert fetched is not None
    assert fetched.state == state
    assert fetched.node == "planner"
    assert fetched.metadata == {"step": 1}


async def test_in_memory_aget_latest_returns_most_recent():
    cp = InMemoryCheckpointer()
    cid_1 = new_checkpoint_id()
    cid_2 = new_checkpoint_id()
    await cp.aput(
        thread_id="t1", checkpoint_id=cid_1, parent_id=None, node="a",
        state={"x": 1},
    )
    await cp.aput(
        thread_id="t1", checkpoint_id=cid_2, parent_id=cid_1, node="b",
        state={"x": 2},
    )
    latest = await cp.aget("t1")
    assert latest is not None
    assert latest.checkpoint_id == cid_2
    assert latest.parent_id == cid_1


async def test_in_memory_alist_newest_first():
    cp = InMemoryCheckpointer()
    ids = [new_checkpoint_id() for _ in range(3)]
    for i, cid in enumerate(ids):
        await cp.aput(
            thread_id="t1", checkpoint_id=cid,
            parent_id=ids[i - 1] if i else None, node=f"n{i}", state={"i": i},
        )
    history = await cp.alist("t1")
    assert [h.checkpoint_id for h in history] == list(reversed(ids))


async def test_in_memory_adelete_removes_thread():
    cp = InMemoryCheckpointer()
    await cp.aput(
        thread_id="t1", checkpoint_id=new_checkpoint_id(),
        parent_id=None, node="a", state={"x": 1},
    )
    await cp.aput(
        thread_id="t1", checkpoint_id=new_checkpoint_id(),
        parent_id=None, node="b", state={"x": 2},
    )
    deleted = await cp.adelete("t1")
    assert deleted == 2
    assert await cp.aget("t1") is None


async def test_in_memory_aput_enforces_blob_cap():
    cp = InMemoryCheckpointer()
    big = {"x": "y" * (MAX_CHECKPOINT_BLOB_BYTES + 16)}
    with pytest.raises(CheckpointBlobTooLarge):
        await cp.aput(
            thread_id="t1", checkpoint_id=new_checkpoint_id(),
            parent_id=None, node="big", state=big,
        )


# -- Sqlite backend (smoke) ---------------------------------------------------


async def test_sqlite_checkpointer_roundtrip(tmp_path):
    pytest.importorskip("aiosqlite", reason="requires eirel[sqlite] extra")
    from eirel.checkpoint.sqlite import SqliteCheckpointer

    cp = SqliteCheckpointer(tmp_path / "ckpt.sqlite")
    try:
        cid = new_checkpoint_id()
        await cp.aput(
            thread_id="t1", checkpoint_id=cid, parent_id=None,
            node="planner", state={"messages": [{"role": "user", "content": "hi"}]},
            metadata={"step": 1},
        )
        latest = await cp.aget("t1")
        assert latest is not None
        assert latest.checkpoint_id == cid
        assert latest.state == {"messages": [{"role": "user", "content": "hi"}]}
        assert latest.metadata == {"step": 1}

        history = await cp.alist("t1")
        assert len(history) == 1
        assert history[0].checkpoint_id == cid

        deleted = await cp.adelete("t1")
        assert deleted == 1
        assert await cp.aget("t1") is None
    finally:
        await cp.aclose()


async def test_sqlite_persists_across_instances(tmp_path):
    pytest.importorskip("aiosqlite", reason="requires eirel[sqlite] extra")
    from eirel.checkpoint.sqlite import SqliteCheckpointer

    db_path = tmp_path / "ckpt.sqlite"
    cp = SqliteCheckpointer(db_path)
    try:
        cid = new_checkpoint_id()
        await cp.aput(
            thread_id="t1", checkpoint_id=cid, parent_id=None,
            node="planner", state={"x": 1},
        )
    finally:
        await cp.aclose()

    cp2 = SqliteCheckpointer(db_path)
    try:
        latest = await cp2.aget("t1")
        assert latest is not None
        assert latest.checkpoint_id == cid
        assert latest.state == {"x": 1}
    finally:
        await cp2.aclose()


# -- Resume token bridge -----------------------------------------------------


def test_resume_token_ttl_constant_matches_token_signing():
    assert RESUME_TOKEN_TTL_SECONDS == 172_800


def test_encode_decode_thread_token_roundtrip():
    secret = "test-secret"
    token = encode_thread_token(
        thread_id="t1", checkpoint_id="abc", secret=secret,
        extra={"validator_hotkey": "5HotkeyX"},
    )
    payload = decode_thread_token(token, secret)
    assert payload["thread_id"] == "t1"
    assert payload["checkpoint_id"] == "abc"
    assert payload["extra"]["validator_hotkey"] == "5HotkeyX"


def test_decode_rejects_tampered_token():
    secret = "test-secret"
    token = encode_thread_token(thread_id="t1", checkpoint_id="abc", secret=secret)
    # Flip the last hex char to a guaranteed-different value (avoid the 1/256
    # case where the original HMAC already ends in the replacement char).
    tampered = token[:-1] + ("1" if token[-1] == "0" else "0")
    with pytest.raises(InvalidResumeToken):
        decode_thread_token(tampered, secret)


def test_decode_rejects_wrong_secret():
    token = encode_thread_token(thread_id="t1", checkpoint_id="abc", secret="a")
    with pytest.raises(InvalidResumeToken):
        decode_thread_token(token, "different")


def test_decode_supports_secret_rotation():
    """Multiple secrets accepted via list — old + new keys both work."""
    token = encode_thread_token(thread_id="t1", checkpoint_id="abc", secret="old")
    payload = decode_thread_token(token, ["new", "old"])
    assert payload["checkpoint_id"] == "abc"


def test_encode_requires_non_empty_ids():
    with pytest.raises(ValueError):
        encode_thread_token(thread_id="", checkpoint_id="abc", secret="s")
    with pytest.raises(ValueError):
        encode_thread_token(thread_id="t1", checkpoint_id="", secret="s")


def test_decode_rejects_payload_missing_required_fields():
    from eirel.token_signing import sign_resume_token

    secret = "s"
    bad_payload = json.dumps({"thread_id": "t1"})  # no checkpoint_id
    bad_token = sign_resume_token(bad_payload, secret)
    with pytest.raises(InvalidResumeToken):
        decode_thread_token(bad_token, secret)
