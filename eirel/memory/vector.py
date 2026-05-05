"""Vector store ABC + adapters for long-term semantic memory.

The ``VectorStore`` interface is intentionally minimal — upsert,
similarity query, delete, namespace per miner. Concrete adapters for
Chroma and Qdrant live behind extras (``eirel[chroma]`` / ``eirel[qdrant]``)
and are imported lazily so the core SDK stays small.

Embeddings are the caller's responsibility. The store accepts pre-
computed vectors so miners can swap embedding models without changing
the store. ``InMemoryVectorStore`` provides a tests/dev backend that
uses cosine similarity.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VectorItem",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaAdapter",
    "QdrantAdapter",
]


@dataclass(frozen=True, slots=True)
class VectorItem:
    """One vector in the store."""

    id: str
    embedding: tuple[float, ...]
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0
    """Populated by query results; ignored on upsert."""


class VectorStore(ABC):
    """Namespace-scoped vector store."""

    @abstractmethod
    async def aupsert(self, namespace: str, items: Sequence[VectorItem]) -> None:
        ...

    @abstractmethod
    async def aquery(
        self,
        namespace: str,
        embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> list[VectorItem]:
        ...

    @abstractmethod
    async def adelete(self, namespace: str, *, ids: Sequence[str] | None = None) -> int:
        """Delete the named items, or the whole namespace when ``ids`` is None."""

    async def aclose(self) -> None:
        return None


# -- In-memory implementation ------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Dictionary-backed vector store with cosine-similarity ranking."""

    def __init__(self) -> None:
        self._namespaces: dict[str, dict[str, VectorItem]] = {}

    async def aupsert(self, namespace: str, items: Sequence[VectorItem]) -> None:
        bucket = self._namespaces.setdefault(namespace, {})
        for item in items:
            bucket[item.id] = item

    async def aquery(
        self,
        namespace: str,
        embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> list[VectorItem]:
        bucket = self._namespaces.get(namespace, {})
        scored: list[VectorItem] = []
        for item in bucket.values():
            score = _cosine(embedding, item.embedding)
            scored.append(
                VectorItem(
                    id=item.id,
                    embedding=item.embedding,
                    text=item.text,
                    metadata=item.metadata,
                    score=score,
                )
            )
        scored.sort(key=lambda v: v.score, reverse=True)
        return scored[:k]

    async def adelete(
        self, namespace: str, *, ids: Sequence[str] | None = None
    ) -> int:
        if ids is None:
            removed = len(self._namespaces.pop(namespace, {}))
            return removed
        bucket = self._namespaces.get(namespace, {})
        removed = 0
        for vid in ids:
            if bucket.pop(vid, None) is not None:
                removed += 1
        return removed


# -- Chroma adapter (extra) --------------------------------------------------


class ChromaAdapter(VectorStore):
    """Chroma adapter — requires ``pip install 'eirel[chroma]'``.

    Wraps a chroma collection per namespace.
    """

    def __init__(self, *, host: str = "localhost", port: int = 8000):
        try:
            import chromadb  # noqa: F401  pragma: no cover
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ChromaAdapter requires the eirel[chroma] extra"
            ) from exc
        import chromadb

        self._client = chromadb.HttpClient(host=host, port=port)

    def _collection(self, namespace: str):
        return self._client.get_or_create_collection(name=namespace)

    async def aupsert(self, namespace: str, items: Sequence[VectorItem]) -> None:
        coll = self._collection(namespace)
        if not items:
            return
        coll.upsert(
            ids=[i.id for i in items],
            embeddings=[list(i.embedding) for i in items],
            documents=[i.text for i in items],
            metadatas=[dict(i.metadata) for i in items],
        )

    async def aquery(
        self,
        namespace: str,
        embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> list[VectorItem]:
        coll = self._collection(namespace)
        result = coll.query(query_embeddings=[list(embedding)], n_results=k)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[VectorItem] = []
        for i, vid in enumerate(ids):
            out.append(
                VectorItem(
                    id=vid,
                    embedding=tuple(),
                    text=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    # Chroma reports L2-style distances; smaller = closer.
                    # Convert to a similarity score for consistency with
                    # cosine-style ranking elsewhere.
                    score=1.0 - float(distances[i]) if i < len(distances) else 0.0,
                )
            )
        return out

    async def adelete(
        self, namespace: str, *, ids: Sequence[str] | None = None
    ) -> int:
        coll = self._collection(namespace)
        if ids is None:
            existing = coll.get()
            existing_ids = existing.get("ids") or []
            if existing_ids:
                coll.delete(ids=existing_ids)
            return len(existing_ids)
        coll.delete(ids=list(ids))
        return len(ids)


# -- Qdrant adapter (extra) --------------------------------------------------


class QdrantAdapter(VectorStore):
    """Qdrant adapter — requires ``pip install 'eirel[qdrant]'``."""

    def __init__(
        self, *, url: str = "http://localhost:6333", api_key: str | None = None
    ):
        try:
            from qdrant_client import AsyncQdrantClient  # noqa: F401  pragma: no cover
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QdrantAdapter requires the eirel[qdrant] extra"
            ) from exc
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(url=url, api_key=api_key)

    async def aupsert(self, namespace: str, items: Sequence[VectorItem]) -> None:
        from qdrant_client.models import PointStruct

        if not items:
            return
        await self._client.upsert(
            collection_name=namespace,
            points=[
                PointStruct(
                    id=i.id,
                    vector=list(i.embedding),
                    payload={"text": i.text, **dict(i.metadata)},
                )
                for i in items
            ],
        )

    async def aquery(
        self,
        namespace: str,
        embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> list[VectorItem]:
        result = await self._client.search(
            collection_name=namespace,
            query_vector=list(embedding),
            limit=k,
        )
        out: list[VectorItem] = []
        for hit in result:
            payload = hit.payload or {}
            text = str(payload.pop("text", "")) if payload else ""
            out.append(
                VectorItem(
                    id=str(hit.id),
                    embedding=tuple(),
                    text=text,
                    metadata=payload,
                    score=float(hit.score),
                )
            )
        return out

    async def adelete(
        self, namespace: str, *, ids: Sequence[str] | None = None
    ) -> int:
        if ids is None:
            await self._client.delete_collection(collection_name=namespace)
            return 0
        await self._client.delete(
            collection_name=namespace, points_selector=list(ids)
        )
        return len(ids)

    async def aclose(self) -> None:
        await self._client.close()
