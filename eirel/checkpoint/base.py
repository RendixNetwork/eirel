"""Checkpointer ABC and shared types.

A checkpointer is a thread-keyed key/value store with parent-pointer
semantics, so a graph run can pause, persist its state, return a resume
token, and continue on a later turn. Different backends (in-memory,
sqlite, postgres-via-HTTP) implement the same ABC; the runtime never
sees the difference.

State blob size is hard-capped at 256 KB. Long conversations must
collapse via :class:`~eirel.memory.summarizer.RollingSummary` (or any
custom reducer) before the cap fires. The cap exists because:

  * Postgres ``BYTEA`` rows live in TOAST when > 8 KB, but the
    serialization cost of large state on every checkpoint is real.
  * Holding the entire chat history in checkpoint state defeats the
    point of memory summarization.
  * Bounded blobs make replay safe — a malicious or buggy miner cannot
    fill the shared store with one giant state dump.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

__all__ = [
    "MAX_CHECKPOINT_BLOB_BYTES",
    "CheckpointBlobTooLarge",
    "CheckpointTuple",
    "Checkpointer",
    "new_checkpoint_id",
    "serialize_state",
    "deserialize_state",
]


MAX_CHECKPOINT_BLOB_BYTES: int = 256 * 1024


class CheckpointBlobTooLarge(ValueError):
    """Raised when an ``aput`` would write a blob over the 256 KB cap."""

    def __init__(self, *, size_bytes: int, limit_bytes: int = MAX_CHECKPOINT_BLOB_BYTES) -> None:
        super().__init__(
            f"checkpoint blob is {size_bytes} bytes; limit is {limit_bytes} bytes "
            f"(use a rolling summary or trim history before persisting)"
        )
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


@dataclass(frozen=True, slots=True)
class CheckpointTuple:
    """One persisted checkpoint.

    ``state`` is the full graph state at the point of capture (post-
    reducer-merge). ``node`` is the name of the node that was about to
    run after this checkpoint was taken — when resuming, the executor
    starts there. ``pending_writes`` carries any partial parallel-branch
    updates not yet merged at the time of pause; the executor folds
    them in before continuing.
    """

    thread_id: str
    checkpoint_id: str
    parent_id: str | None
    created_at: datetime
    node: str | None
    state: Mapping[str, Any]
    pending_writes: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def new_checkpoint_id() -> str:
    """Generate a fresh, opaque checkpoint id (uuid4 hex, 32 chars)."""
    return uuid4().hex


def serialize_state(state: Mapping[str, Any]) -> bytes:
    """Serialize state to bytes, enforcing the 256 KB cap.

    Uses compact JSON. Bytes-keyed values raise — checkpoints must be
    pure JSON-roundtrip-safe so any backend (sqlite/postgres/redis) can
    store them without bespoke encoders.
    """
    blob = json.dumps(state, separators=(",", ":"), default=_json_default).encode("utf-8")
    if len(blob) > MAX_CHECKPOINT_BLOB_BYTES:
        raise CheckpointBlobTooLarge(size_bytes=len(blob))
    return blob


def deserialize_state(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _json_default(value: Any) -> Any:
    """Best-effort JSON fallback for graph state.

    Pydantic models go through ``model_dump``; sets/tuples become lists;
    datetimes become isoformat strings. Anything else raises so the
    miner can spot non-serializable state early.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(
        f"checkpoint state contains non-JSON-serializable value: "
        f"{type(value).__name__}"
    )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Checkpointer(ABC):
    """Persistent key/value store keyed by ``(thread_id, checkpoint_id)``.

    Implementations:

      * :class:`~eirel.checkpoint.memory.InMemoryCheckpointer` — tests + dev
      * :class:`~eirel.checkpoint.sqlite.SqliteCheckpointer` — single-process
        durability behind ``eirel[sqlite]``
      * :class:`~eirel.checkpoint.postgres.PostgresCheckpointer` —
        HTTP-write-through to the eirel-ai control plane (no DB creds in
        the miner image), behind ``eirel[postgres]``

    Backends MUST:

      * Honor ``MAX_CHECKPOINT_BLOB_BYTES`` (raise ``CheckpointBlobTooLarge``).
      * Treat ``aput`` as immutable: never mutate an existing
        ``checkpoint_id``. Always create a new id and link via
        ``parent_id``.
      * Make ``aget(thread_id, None)`` return the latest checkpoint
        (most recent ``created_at``).
    """

    @abstractmethod
    async def aput(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        parent_id: str | None,
        node: str | None,
        state: Mapping[str, Any],
        pending_writes: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointTuple:
        ...

    @abstractmethod
    async def aget(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> CheckpointTuple | None:
        """Return the requested checkpoint, or the latest when ``checkpoint_id`` is None."""

    @abstractmethod
    async def alist(
        self,
        thread_id: str,
        *,
        limit: int = 50,
    ) -> list[CheckpointTuple]:
        """Return checkpoints for the thread, newest first."""

    @abstractmethod
    async def adelete(self, thread_id: str) -> int:
        """Delete every checkpoint for ``thread_id``. Returns the count removed."""

    async def aclose(self) -> None:
        """Optional: release any backend resources (DB connections, HTTP clients)."""
        return None
