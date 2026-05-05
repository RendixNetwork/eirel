"""Tests for the safety guard primitives + runtime integration."""
from __future__ import annotations

import pytest

from eirel import (
    ChainedGuard,
    Guard,
    GuardVerdict,
    NoopGuard,
    StateField,
    StateGraph,
    StateSpec,
    add_messages,
    replace,
)
from eirel.graph import END


# -- ChainedGuard ------------------------------------------------------------


class _AllowGuard(Guard):
    def __init__(self, *, redactions: dict | None = None, label: str = "allow"):
        self._redactions = redactions
        self._label = label

    async def pre_input(self, state):
        return GuardVerdict(allow=True, redactions=self._redactions, metadata={"label": self._label})

    async def post_output(self, state):
        return GuardVerdict(allow=True, redactions=self._redactions, metadata={"label": self._label})


class _DenyGuard(Guard):
    def __init__(self, *, reason: str = "no", label: str = "deny"):
        self._reason = reason
        self._label = label

    async def pre_input(self, state):
        return GuardVerdict.deny(self._reason, label=self._label)

    async def post_output(self, state):
        return GuardVerdict.deny(self._reason, label=self._label)


async def test_chained_guard_allow_when_all_allow():
    chain = ChainedGuard([_AllowGuard(label="a"), _AllowGuard(label="b")])
    verdict = await chain.pre_input({})
    assert verdict.allow is True
    assert "chain" in verdict.metadata


async def test_chained_guard_short_circuits_on_first_deny():
    chain = ChainedGuard([_AllowGuard(label="a"), _DenyGuard(reason="bad"), _AllowGuard(label="c")])
    verdict = await chain.pre_input({})
    assert verdict.allow is False
    assert "bad" in (verdict.reason or "")
    assert verdict.metadata["denied_by_index"] == 1
    # Index 2 ("c") never ran — it's not in the chain.
    chain_meta = verdict.metadata["chain"]
    assert "guard_2" not in chain_meta


async def test_chained_guard_merges_redactions_left_to_right():
    chain = ChainedGuard([
        _AllowGuard(redactions={"a": 1}, label="first"),
        _AllowGuard(redactions={"a": 2, "b": 3}, label="second"),
    ])
    verdict = await chain.pre_input({})
    assert verdict.allow is True
    assert verdict.redactions == {"a": 2, "b": 3}


def test_chained_guard_rejects_empty():
    with pytest.raises(ValueError):
        ChainedGuard([])


# -- Runtime integration -----------------------------------------------------


def _trivial_graph(spec):
    async def respond(state):
        return {"messages": {"role": "assistant", "content": "ok"}}

    g = StateGraph(spec)
    g.add_node("respond", respond)
    g.add_edge("respond", END)
    g.set_entry_point("respond")
    return g


@pytest.fixture
def spec():
    return StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
        "redaction_marker": StateField(reducer=replace, default=""),
    })


async def test_pre_input_deny_yields_failed_done(spec):
    g = _trivial_graph(spec)
    compiled = g.compile(safety=_DenyGuard(reason="forbidden_topic"))
    chunks = []
    async for chunk in compiled.astream(spec.init()):
        chunks.append(chunk)
    assert chunks[-1].event == "done"
    assert chunks[-1].status == "failed"
    assert "forbidden_topic" in (chunks[-1].error or "")
    assert chunks[-1].metadata["guard"]["stage"] == "pre_input"


async def test_post_output_deny_blocks_completion(spec):
    """Pre-input allows, but post_output denies — caller must see failed."""
    class _OneSidedGuard(Guard):
        async def pre_input(self, state):
            return GuardVerdict.ok()

        async def post_output(self, state):
            return GuardVerdict.deny("output_unsafe")

    g = _trivial_graph(spec)
    compiled = g.compile(safety=_OneSidedGuard())
    chunks = []
    async for chunk in compiled.astream(spec.init()):
        chunks.append(chunk)
    assert chunks[-1].status == "failed"
    assert chunks[-1].metadata["guard"]["stage"] == "post_output"


async def test_pre_input_redactions_merge_into_state(spec):
    """A guard that redacts replaces field values via the spec's reducers."""
    class _RedactingGuard(Guard):
        async def pre_input(self, state):
            return GuardVerdict(allow=True, redactions={"redaction_marker": "REDACTED"})

        async def post_output(self, state):
            return GuardVerdict.ok()

    captured = {}

    async def respond(state):
        captured["marker"] = state["redaction_marker"]
        return None

    g = StateGraph(spec)
    g.add_node("respond", respond)
    g.add_edge("respond", END)
    g.set_entry_point("respond")
    compiled = g.compile(safety=_RedactingGuard())
    await compiled.ainvoke(spec.init(redaction_marker="ORIGINAL"))
    assert captured["marker"] == "REDACTED"


async def test_noop_guard_does_not_block_anything(spec):
    g = _trivial_graph(spec)
    compiled = g.compile(safety=NoopGuard())
    final = await compiled.ainvoke(spec.init())
    assert final["messages"][-1]["content"] == "ok"
