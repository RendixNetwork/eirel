"""In-process checkpointer for tests + local development.

Not durable — process restart loses everything. Production miners
should use ``SqliteCheckpointer`` (single pod) or ``PostgresCheckpointer``
(hosted runtime, HTTP write-through).
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from eirel.checkpoint.base import (
    CheckpointTuple,
    Checkpointer,
    serialize_state,
    utcnow,
)

__all__ = ["InMemoryCheckpointer"]


class InMemoryCheckpointer(Checkpointer):
    """Dict-backed checkpointer. Thread-safe via an ``asyncio.Lock``."""

    def __init__(self) -> None:
        self._store: dict[str, list[CheckpointTuple]] = {}
        self._lock = asyncio.Lock()

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
        # Enforce the 256 KB cap; the serialized form is what backends
        # would persist, so we validate against the real bytes.
        serialize_state(dict(state))
        tup = CheckpointTuple(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            parent_id=parent_id,
            created_at=utcnow(),
            node=node,
            state=dict(state),
            pending_writes=tuple(dict(w) for w in pending_writes),
            metadata=dict(metadata or {}),
        )
        async with self._lock:
            self._store.setdefault(thread_id, []).append(tup)
        return tup

    async def aget(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> CheckpointTuple | None:
        async with self._lock:
            entries = self._store.get(thread_id, [])
            if not entries:
                return None
            if checkpoint_id is None:
                return entries[-1]
            for tup in reversed(entries):
                if tup.checkpoint_id == checkpoint_id:
                    return tup
        return None

    async def alist(
        self,
        thread_id: str,
        *,
        limit: int = 50,
    ) -> list[CheckpointTuple]:
        async with self._lock:
            entries = list(reversed(self._store.get(thread_id, [])))
            return entries[:limit]

    async def adelete(self, thread_id: str) -> int:
        async with self._lock:
            entries = self._store.pop(thread_id, [])
            return len(entries)
