"""Memory primitives for graph agents.

  * :class:`~eirel.memory.summarizer.RollingSummary` collapses
    long chat history into a single summary entry every N turns to keep
    checkpoint blobs under the 256 KB cap.
  * :class:`~eirel.memory.vector.VectorStore` is the ABC for long-term
    semantic memory. ``ChromaAdapter`` and ``QdrantAdapter`` are
    behind extras.
"""
from __future__ import annotations

from eirel.memory.summarizer import RollingSummary
from eirel.memory.vector import (
    InMemoryVectorStore,
    VectorItem,
    VectorStore,
)


def __getattr__(name: str):
    if name == "ChromaAdapter":
        from eirel.memory.vector import ChromaAdapter
        return ChromaAdapter
    if name == "QdrantAdapter":
        from eirel.memory.vector import QdrantAdapter
        return QdrantAdapter
    raise AttributeError(f"module 'eirel.memory' has no attribute {name!r}")


__all__ = [
    "ChromaAdapter",
    "InMemoryVectorStore",
    "QdrantAdapter",
    "RollingSummary",
    "VectorItem",
    "VectorStore",
]
