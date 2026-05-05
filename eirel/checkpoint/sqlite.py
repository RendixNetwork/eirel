"""SQLite-backed checkpointer.

Single-process durability for graph runs. Survives pod restarts when
mounted on a persistent volume; survives nothing when the file lives in
container-local ephemeral storage. Intended for single-replica miner
pods where the owner-api can mount a PVC at the same path each time.

Requires the ``aiosqlite`` extra (``pip install eirel[sqlite]``).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eirel.checkpoint.base import (
    CheckpointTuple,
    Checkpointer,
    deserialize_state,
    serialize_state,
)

__all__ = ["SqliteCheckpointer"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    created_at TEXT NOT NULL,
    node TEXT,
    state_blob BLOB NOT NULL,
    pending_writes TEXT NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS ix_checkpoints_thread_created
    ON checkpoints(thread_id, created_at DESC);
"""


def _row_to_tuple(row: tuple[Any, ...]) -> CheckpointTuple:
    (
        thread_id,
        checkpoint_id,
        parent_id,
        created_at_iso,
        node,
        state_blob,
        pending_writes_json,
        metadata_json,
    ) = row
    return CheckpointTuple(
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        parent_id=parent_id,
        created_at=datetime.fromisoformat(created_at_iso),
        node=node,
        state=deserialize_state(state_blob if isinstance(state_blob, bytes) else bytes(state_blob)),
        pending_writes=tuple(json.loads(pending_writes_json or "[]")),
        metadata=json.loads(metadata_json or "{}"),
    )


class SqliteCheckpointer(Checkpointer):
    """SQLite-backed checkpointer using ``aiosqlite``.

    Lazy schema migration on first use. Connection is held open for the
    lifetime of the checkpointer; ``aclose`` releases it.
    """

    def __init__(self, path: str | Path):
        try:
            import aiosqlite  # noqa: F401  pragma: no cover — import-time check
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SqliteCheckpointer requires the eirel[sqlite] extra: pip install 'eirel[sqlite]'"
            ) from exc
        self._path = str(path)
        self._conn: Any = None  # aiosqlite.Connection
        self._initialized = False

    async def _get_conn(self) -> Any:
        import aiosqlite

        if self._conn is None:
            self._conn = await aiosqlite.connect(self._path)
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
            self._initialized = True
        return self._conn

    async def aclose(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

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
        blob = serialize_state(dict(state))
        created_at = datetime.now(timezone.utc)
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT INTO checkpoints
                (thread_id, checkpoint_id, parent_id, created_at, node,
                 state_blob, pending_writes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                checkpoint_id,
                parent_id,
                created_at.isoformat(),
                node,
                blob,
                json.dumps([dict(w) for w in pending_writes], separators=(",", ":")),
                json.dumps(dict(metadata or {}), separators=(",", ":")),
            ),
        )
        await conn.commit()
        return CheckpointTuple(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            parent_id=parent_id,
            created_at=created_at,
            node=node,
            state=dict(state),
            pending_writes=tuple(dict(w) for w in pending_writes),
            metadata=dict(metadata or {}),
        )

    async def aget(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> CheckpointTuple | None:
        conn = await self._get_conn()
        if checkpoint_id is None:
            cursor = await conn.execute(
                "SELECT thread_id, checkpoint_id, parent_id, created_at, node, "
                "state_blob, pending_writes, metadata "
                "FROM checkpoints WHERE thread_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT thread_id, checkpoint_id, parent_id, created_at, node, "
                "state_blob, pending_writes, metadata "
                "FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id),
            )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        if row is None:
            return None
        return _row_to_tuple(row)

    async def alist(
        self,
        thread_id: str,
        *,
        limit: int = 50,
    ) -> list[CheckpointTuple]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT thread_id, checkpoint_id, parent_id, created_at, node, "
            "state_blob, pending_writes, metadata "
            "FROM checkpoints WHERE thread_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (thread_id, limit),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_tuple(r) for r in rows]

    async def adelete(self, thread_id: str) -> int:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        try:
            count = cursor.rowcount or 0
        finally:
            await cursor.close()
        await conn.commit()
        return int(count)
