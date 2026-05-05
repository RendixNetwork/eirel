"""Tests for RetryPolicy + FallbackChain on GeneralChatToolCatalog."""
from __future__ import annotations

import time
from typing import Any

import pytest

from eirel.families.general_chat.budget import INSTANT_BUDGET, BudgetTracker
from eirel.families.general_chat.response import TraceRecorder
from eirel.families.general_chat.tools import (
    FallbackChain,
    GeneralChatTool,
    GeneralChatToolCatalog,
    RetryPolicy,
)


class _FlakyTool(GeneralChatTool):
    """Fails for the first ``failures`` calls, then succeeds."""

    def __init__(self, name: str, *, failures: int, exc: type[BaseException] = RuntimeError):
        self._name = name
        self._failures = failures
        self._exc = exc
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "flaky"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        if self.call_count <= self._failures:
            raise self._exc(f"transient {self.call_count}")
        return {"ok": True, "attempts": self.call_count}


class _StaticTool(GeneralChatTool):
    def __init__(self, name: str, payload: Any) -> None:
        self._name = name
        self._payload = payload

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "static"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, **kwargs: Any) -> Any:
        return self._payload


def _catalog(tools: list[GeneralChatTool]) -> GeneralChatToolCatalog:
    return GeneralChatToolCatalog(
        tools, budget=BudgetTracker(budget=INSTANT_BUDGET), trace=TraceRecorder()
    )


# -- RetryPolicy --------------------------------------------------------------


def test_retry_policy_validates_max_attempts():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_retry_policy_validates_backoff_kind():
    with pytest.raises(ValueError):
        RetryPolicy(backoff="quadratic")


def test_retry_policy_linear_backoff_seconds():
    p = RetryPolicy(max_attempts=3, backoff_seconds=0.1, backoff="linear")
    assert p.sleep_seconds(1) == 0
    assert p.sleep_seconds(2) == pytest.approx(0.1)
    assert p.sleep_seconds(3) == pytest.approx(0.2)


def test_retry_policy_exponential_backoff_seconds():
    p = RetryPolicy(max_attempts=4, backoff_seconds=0.1, backoff="exponential")
    assert p.sleep_seconds(1) == 0
    assert p.sleep_seconds(2) == pytest.approx(0.1)
    assert p.sleep_seconds(3) == pytest.approx(0.2)
    assert p.sleep_seconds(4) == pytest.approx(0.4)


async def test_execute_with_policy_recovers_from_transient_failures():
    tool = _FlakyTool("flaky", failures=1)
    catalog = _catalog([tool])
    policy = RetryPolicy(max_attempts=2)
    result = await catalog.execute_with_policy("flaky", policy=policy, q="hi")
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert tool.call_count == 2


async def test_execute_with_policy_re_raises_after_attempts_exhausted():
    tool = _FlakyTool("flaky", failures=5)
    catalog = _catalog([tool])
    policy = RetryPolicy(max_attempts=2)
    with pytest.raises(RuntimeError, match="transient 2"):
        await catalog.execute_with_policy("flaky", policy=policy, q="hi")
    assert tool.call_count == 2


async def test_execute_with_policy_does_not_retry_unmatched_exception():
    tool = _FlakyTool("flaky", failures=5, exc=ValueError)
    catalog = _catalog([tool])
    policy = RetryPolicy(max_attempts=3, retry_on=(RuntimeError,))
    with pytest.raises(ValueError):
        await catalog.execute_with_policy("flaky", policy=policy, q="hi")
    # Only one call — ValueError is not in retry_on.
    assert tool.call_count == 1


async def test_execute_with_policy_records_attempts_in_trace():
    tool = _FlakyTool("flaky", failures=1)
    trace = TraceRecorder()
    catalog = GeneralChatToolCatalog(
        [tool], budget=BudgetTracker(budget=INSTANT_BUDGET), trace=trace
    )
    policy = RetryPolicy(max_attempts=2)
    await catalog.execute_with_policy("flaky", policy=policy, q="hi")
    names = [tc.tool_name for tc in trace.tool_calls]
    # Attempt 1 failed → recorded as attempt_1; attempt 2 succeeded → also attempt_2.
    assert "flaky:attempt_1" in names
    assert "flaky:attempt_2" in names


async def test_execute_with_policy_unknown_tool_raises():
    catalog = _catalog([_StaticTool("known", {"v": 1})])
    with pytest.raises(ValueError, match="unknown"):
        await catalog.execute_with_policy(
            "unknown", policy=RetryPolicy(max_attempts=2)
        )


# -- FallbackChain ------------------------------------------------------------


async def test_fallback_chain_returns_primary_when_non_empty():
    catalog = _catalog([
        _StaticTool("primary", {"results": ["a", "b"]}),
        _StaticTool("backup", {"results": ["c"]}),
    ])
    chain = FallbackChain(primary="primary", fallbacks=("backup",))
    result = await catalog.execute_chain(chain)
    assert result == {"results": ["a", "b"]}


async def test_fallback_chain_falls_through_on_empty_primary():
    catalog = _catalog([
        _StaticTool("primary", {"results": []}),  # empty
        _StaticTool("backup", {"results": ["recovered"]}),
    ])
    chain = FallbackChain(primary="primary", fallbacks=("backup",))
    result = await catalog.execute_chain(chain)
    assert result == {"results": ["recovered"]}


async def test_fallback_chain_falls_through_on_primary_exception():
    catalog = _catalog([
        _FlakyTool("primary", failures=1),  # raises once
        _StaticTool("backup", {"results": ["ok"]}),
    ])
    chain = FallbackChain(primary="primary", fallbacks=("backup",))
    result = await catalog.execute_chain(chain)
    # Primary raised once, no retry policy — falls to backup.
    assert result == {"results": ["ok"]}


async def test_fallback_chain_with_retry_policy_retries_each_link():
    primary = _FlakyTool("primary", failures=5)  # always fails
    backup = _FlakyTool("backup", failures=1)  # succeeds on 2nd try
    catalog = _catalog([primary, backup])
    chain = FallbackChain(
        primary="primary",
        fallbacks=("backup",),
        retry_policy=RetryPolicy(max_attempts=2),
    )
    result = await catalog.execute_chain(chain)
    assert result["ok"] is True
    assert primary.call_count == 2  # tried twice, both failed
    assert backup.call_count == 2  # tried twice, succeeded on second


async def test_fallback_chain_returns_last_empty_when_all_empty():
    catalog = _catalog([
        _StaticTool("a", {"results": []}),
        _StaticTool("b", {"results": []}),
    ])
    chain = FallbackChain(primary="a", fallbacks=("b",))
    result = await catalog.execute_chain(chain)
    assert result == {"results": []}


async def test_fallback_chain_re_raises_when_all_fail():
    catalog = _catalog([
        _FlakyTool("a", failures=5),
        _FlakyTool("b", failures=5),
    ])
    chain = FallbackChain(primary="a", fallbacks=("b",))
    with pytest.raises(RuntimeError):
        await catalog.execute_chain(chain)


async def test_fallback_chain_custom_predicate():
    catalog = _catalog([
        _StaticTool("a", {"score": 0.3}),
        _StaticTool("b", {"score": 0.9}),
    ])
    chain = FallbackChain(
        primary="a",
        fallbacks=("b",),
        predicate=lambda r: isinstance(r, dict) and r.get("score", 0) > 0.5,
    )
    result = await catalog.execute_chain(chain)
    assert result == {"score": 0.9}


async def test_fallback_chain_records_fallback_step_in_trace():
    trace = TraceRecorder()
    catalog = GeneralChatToolCatalog(
        [
            _StaticTool("a", {"results": []}),
            _StaticTool("b", {"results": ["x"]}),
        ],
        budget=BudgetTracker(budget=INSTANT_BUDGET),
        trace=trace,
    )
    chain = FallbackChain(primary="a", fallbacks=("b",))
    await catalog.execute_chain(chain)
    names = [tc.tool_name for tc in trace.tool_calls]
    assert "a" in names
    assert "b:fallback_1" in names


async def test_retry_policy_zero_backoff_does_not_sleep():
    """Sanity: zero backoff means retry is immediate."""
    tool = _FlakyTool("flaky", failures=2)
    catalog = _catalog([tool])
    t0 = time.monotonic()
    await catalog.execute_with_policy(
        "flaky",
        policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
        q="hi",
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, f"unexpected sleep with zero backoff: {elapsed:.3f}s"
