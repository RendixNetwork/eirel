"""Tests for the rolling summarizer + in-memory vector store."""
from __future__ import annotations

import pytest

from eirel.memory import InMemoryVectorStore, RollingSummary, VectorItem


# -- RollingSummary -----------------------------------------------------------


async def test_rolling_summary_no_op_when_under_threshold():
    async def summarize(messages):
        raise AssertionError("should not be called")

    rolling = RollingSummary(summarizer=summarize, every_n_turns=10, keep_recent=2)
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    out = await rolling.maybe_summarize(msgs)
    assert out is None


async def test_rolling_summary_collapses_when_over_threshold():
    captured: list[list[dict]] = []

    async def summarize(messages):
        captured.append(messages)
        return "compressed"

    rolling = RollingSummary(summarizer=summarize, every_n_turns=5, keep_recent=2)
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    out = await rolling.maybe_summarize(msgs)
    assert out is not None
    # Result: one [summary] system message + the 2 most-recent.
    assert len(out) == 3
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("[summary] ")
    assert "compressed" in out[0]["content"]
    # The two trailing messages are the original last two.
    assert out[1] == msgs[-2]
    assert out[2] == msgs[-1]
    # Summarizer saw everything except the kept-recent suffix.
    assert captured[0] == msgs[:-2]


def test_rolling_summary_validates_constructor_args():
    async def s(_):
        return ""

    with pytest.raises(ValueError):
        RollingSummary(summarizer=s, every_n_turns=1, keep_recent=1)
    with pytest.raises(ValueError):
        RollingSummary(summarizer=s, every_n_turns=5, keep_recent=0)
    with pytest.raises(ValueError):
        RollingSummary(summarizer=s, every_n_turns=5, keep_recent=5)


# -- InMemoryVectorStore -----------------------------------------------------


async def test_vector_store_upsert_and_query():
    store = InMemoryVectorStore()
    items = [
        VectorItem(id="a", embedding=(1.0, 0.0, 0.0), text="apple"),
        VectorItem(id="b", embedding=(0.0, 1.0, 0.0), text="banana"),
        VectorItem(id="c", embedding=(0.9, 0.1, 0.0), text="apricot"),
    ]
    await store.aupsert("ns", items)
    results = await store.aquery("ns", (1.0, 0.0, 0.0), k=2)
    assert [r.id for r in results] == ["a", "c"]
    # Cosine similarity puts a (perfect alignment) first.
    assert results[0].score == pytest.approx(1.0, rel=1e-3)


async def test_vector_store_namespaces_are_isolated():
    store = InMemoryVectorStore()
    await store.aupsert(
        "ns1", [VectorItem(id="a", embedding=(1.0, 0.0), text="ns1")]
    )
    await store.aupsert(
        "ns2", [VectorItem(id="a", embedding=(0.0, 1.0), text="ns2")]
    )
    ns1 = await store.aquery("ns1", (1.0, 0.0), k=5)
    ns2 = await store.aquery("ns2", (1.0, 0.0), k=5)
    assert ns1[0].text == "ns1"
    assert ns2[0].text == "ns2"


async def test_vector_store_delete_by_id():
    store = InMemoryVectorStore()
    await store.aupsert(
        "ns",
        [
            VectorItem(id="a", embedding=(1.0,), text="x"),
            VectorItem(id="b", embedding=(1.0,), text="y"),
        ],
    )
    deleted = await store.adelete("ns", ids=["a"])
    assert deleted == 1
    remaining = await store.aquery("ns", (1.0,), k=5)
    assert {r.id for r in remaining} == {"b"}


async def test_vector_store_delete_namespace():
    store = InMemoryVectorStore()
    await store.aupsert(
        "ns",
        [VectorItem(id=str(i), embedding=(float(i),), text="x") for i in range(3)],
    )
    deleted = await store.adelete("ns")
    assert deleted == 3
    assert await store.aquery("ns", (1.0,), k=5) == []
