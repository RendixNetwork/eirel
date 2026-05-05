"""Persistent state for graph runs.

See :mod:`eirel.checkpoint.base` for the ``Checkpointer`` ABC contract.
"""
from __future__ import annotations

from eirel.checkpoint.base import (
    MAX_CHECKPOINT_BLOB_BYTES,
    CheckpointBlobTooLarge,
    CheckpointTuple,
    Checkpointer,
    deserialize_state,
    new_checkpoint_id,
    serialize_state,
)
from eirel.checkpoint.memory import InMemoryCheckpointer
from eirel.checkpoint.resume import (
    InvalidResumeToken,
    RESUME_TOKEN_TTL_SECONDS,
    decode_thread_token,
    encode_thread_token,
)


def _lazy(name: str):
    if name == "SqliteCheckpointer":
        from eirel.checkpoint.sqlite import SqliteCheckpointer
        return SqliteCheckpointer
    if name == "PostgresCheckpointer":
        from eirel.checkpoint.postgres import PostgresCheckpointer
        return PostgresCheckpointer
    raise AttributeError(f"module 'eirel.checkpoint' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy(name)


__all__ = [
    "MAX_CHECKPOINT_BLOB_BYTES",
    "CheckpointBlobTooLarge",
    "CheckpointTuple",
    "Checkpointer",
    "InMemoryCheckpointer",
    "InvalidResumeToken",
    "RESUME_TOKEN_TTL_SECONDS",
    "PostgresCheckpointer",
    "SqliteCheckpointer",
    "decode_thread_token",
    "deserialize_state",
    "encode_thread_token",
    "new_checkpoint_id",
    "serialize_state",
]
