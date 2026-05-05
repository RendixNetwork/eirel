"""HTTP-write-through checkpointer for the hosted runtime.

The miner pod **does not** hold Postgres credentials. Instead it POSTs
checkpoints to eirel-ai's control-plane endpoints; eirel-ai owns the
shared Postgres and enforces tenant isolation by ``deployment_id``
namespace.

Endpoints (eirel-ai side):
    POST /v1/internal/checkpoints/{thread_id}             — write
    GET  /v1/internal/checkpoints/{thread_id}/latest      — read latest
    GET  /v1/internal/checkpoints/{thread_id}/history     — list

Auth: ``EIREL_CHECKPOINT_BACKEND_TOKEN`` is the bearer token for the
internal-service-only routes. Owner-api stamps it into the pod env
when deploying graph-runtime miners.

The "Postgres" in the name describes the *backing store* (a shared
Postgres on the eirel-ai side), not the miner's connection. The miner
sees an HTTP API; that's the entire surface.
"""
from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import httpx

from eirel.checkpoint.base import (
    CheckpointTuple,
    Checkpointer,
    serialize_state,
)

__all__ = ["PostgresCheckpointer"]


def _decode_state(payload: Any) -> dict[str, Any]:
    """Accept either a base64 blob or an inline JSON object for the state."""
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        # Base64 → JSON.
        try:
            decoded = base64.b64decode(payload.encode("ascii"))
            return json.loads(decoded.decode("utf-8"))
        except Exception:  # noqa: BLE001
            # Maybe it's a plain JSON string.
            return json.loads(payload)
    raise TypeError(f"unexpected state payload type: {type(payload).__name__}")


def _from_response(body: Mapping[str, Any]) -> CheckpointTuple:
    return CheckpointTuple(
        thread_id=str(body["thread_id"]),
        checkpoint_id=str(body["checkpoint_id"]),
        parent_id=body.get("parent_id"),
        created_at=datetime.fromisoformat(str(body["created_at"])),
        node=body.get("node"),
        state=_decode_state(body.get("state") or {}),
        pending_writes=tuple(body.get("pending_writes") or []),
        metadata=dict(body.get("metadata") or {}),
    )


class PostgresCheckpointer(Checkpointer):
    """Posts checkpoints to eirel-ai over HTTP; never touches Postgres directly."""

    def __init__(
        self,
        *,
        backend_url: str | None = None,
        namespace: str | None = None,
        bearer_token: str | None = None,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.backend_url = (
            backend_url or os.getenv("EIREL_CHECKPOINT_BACKEND_URL", "")
        ).rstrip("/")
        if not self.backend_url:
            raise RuntimeError(
                "PostgresCheckpointer requires EIREL_CHECKPOINT_BACKEND_URL "
                "or backend_url= argument"
            )
        self.namespace = namespace or os.getenv("EIREL_CHECKPOINT_NAMESPACE", "")
        self.bearer_token = bearer_token or os.getenv("EIREL_CHECKPOINT_BACKEND_TOKEN", "")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if self.namespace:
            headers["X-Eirel-Checkpoint-Namespace"] = self.namespace
        return headers

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
        # Locally enforce the 256 KB cap before sending — fail fast,
        # don't waste a round-trip on a doomed write.
        blob = serialize_state(dict(state))
        encoded = base64.b64encode(blob).decode("ascii")
        body = {
            "checkpoint_id": checkpoint_id,
            "parent_id": parent_id,
            "node": node,
            "state": encoded,
            "pending_writes": [dict(w) for w in pending_writes],
            "metadata": dict(metadata or {}),
        }
        client = await self._get_client()
        resp = await client.post(
            f"{self.backend_url}/v1/internal/checkpoints/{thread_id}",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return _from_response(resp.json())

    async def aget(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> CheckpointTuple | None:
        client = await self._get_client()
        if checkpoint_id is None:
            url = f"{self.backend_url}/v1/internal/checkpoints/{thread_id}/latest"
            params: dict[str, str] = {}
        else:
            url = f"{self.backend_url}/v1/internal/checkpoints/{thread_id}/history"
            params = {"checkpoint_id": checkpoint_id, "limit": "1"}
        resp = await client.get(url, params=params, headers=self._headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        if checkpoint_id is None:
            if not body:
                return None
            return _from_response(body)
        items = body.get("items") or []
        if not items:
            return None
        return _from_response(items[0])

    async def alist(
        self,
        thread_id: str,
        *,
        limit: int = 50,
    ) -> list[CheckpointTuple]:
        client = await self._get_client()
        url = f"{self.backend_url}/v1/internal/checkpoints/{thread_id}/history"
        resp = await client.get(
            url,
            params={"limit": str(limit)},
            headers=self._headers(),
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        body = resp.json()
        return [_from_response(item) for item in body.get("items") or []]

    async def adelete(self, thread_id: str) -> int:
        client = await self._get_client()
        url = f"{self.backend_url}/v1/internal/checkpoints/{thread_id}"
        resp = await client.delete(url, headers=self._headers())
        if resp.status_code == 404:
            return 0
        resp.raise_for_status()
        body = resp.json()
        return int(body.get("deleted", 0))
