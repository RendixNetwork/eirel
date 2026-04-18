from __future__ import annotations

import pytest

from eirel.families.general_chat.budget import (
    INSTANT_BUDGET,
    INSTANT_WEB_SEARCH_BUDGET,
    THINKING_BUDGET,
    THINKING_WEB_SEARCH_BUDGET,
    BudgetExhaustedError,
    BudgetTracker,
    ModeBudget,
    RunBudget,
    RunBudgetExhaustedError,
    RunCostTracker,
    get_budget,
)


# -- Mode budget constants ----------------------------------------------------


def test_instant_budget_constants():
    assert INSTANT_BUDGET.mode == "instant"
    assert INSTANT_BUDGET.web_search is False
    assert INSTANT_BUDGET.latency_seconds == 15.0
    assert INSTANT_BUDGET.output_tokens == 1024
    assert INSTANT_BUDGET.reasoning_tokens == 0


def test_instant_web_search_budget_constants():
    assert INSTANT_WEB_SEARCH_BUDGET.mode == "instant"
    assert INSTANT_WEB_SEARCH_BUDGET.web_search is True
    assert INSTANT_WEB_SEARCH_BUDGET.latency_seconds == 20.0
    assert INSTANT_WEB_SEARCH_BUDGET.output_tokens == 1024
    assert INSTANT_WEB_SEARCH_BUDGET.reasoning_tokens == 0


def test_thinking_budget_constants():
    assert THINKING_BUDGET.mode == "thinking"
    assert THINKING_BUDGET.web_search is False
    assert THINKING_BUDGET.latency_seconds == 60.0
    assert THINKING_BUDGET.output_tokens == 4096
    assert THINKING_BUDGET.reasoning_tokens == 16384


def test_thinking_web_search_budget_constants():
    assert THINKING_WEB_SEARCH_BUDGET.mode == "thinking"
    assert THINKING_WEB_SEARCH_BUDGET.web_search is True
    assert THINKING_WEB_SEARCH_BUDGET.latency_seconds == 75.0
    assert THINKING_WEB_SEARCH_BUDGET.output_tokens == 4096
    assert THINKING_WEB_SEARCH_BUDGET.reasoning_tokens == 16384


def test_mode_budget_is_frozen():
    with pytest.raises(AttributeError):
        INSTANT_BUDGET.latency_seconds = 99.0  # type: ignore[misc]


# -- get_budget ---------------------------------------------------------------


def test_get_budget_instant_no_web():
    assert get_budget("instant", web_search=False) is INSTANT_BUDGET


def test_get_budget_instant_with_web():
    assert get_budget("instant", web_search=True) is INSTANT_WEB_SEARCH_BUDGET


def test_get_budget_thinking_no_web():
    assert get_budget("thinking", web_search=False) is THINKING_BUDGET


def test_get_budget_thinking_with_web():
    assert get_budget("thinking", web_search=True) is THINKING_WEB_SEARCH_BUDGET


def test_get_budget_invalid_mode():
    with pytest.raises(ValueError, match="unknown chat mode"):
        get_budget("turbo", web_search=False)  # type: ignore[arg-type]


# -- BudgetTracker ------------------------------------------------------------


def _make_tracker(budget: ModeBudget, *, clock_value: list[float] | None = None) -> BudgetTracker:
    if clock_value is None:
        clock_value = [0.0]
    return BudgetTracker(budget=budget, _clock=lambda: clock_value[0])


def test_tracker_initial_state():
    clock = [0.0]
    tracker = _make_tracker(INSTANT_BUDGET, clock_value=clock)
    assert tracker.elapsed_seconds == 0
    assert tracker.output_tokens_used == 0
    assert tracker.reasoning_tokens_used == 0
    tracker.check()


def test_tracker_output_token_exhaustion_instant():
    tracker = _make_tracker(INSTANT_BUDGET)
    tracker.record_output_tokens(1024)
    with pytest.raises(BudgetExhaustedError, match="output_tokens"):
        tracker.record_output_tokens(1)


def test_tracker_output_token_exhaustion_thinking():
    tracker = _make_tracker(THINKING_BUDGET)
    tracker.record_output_tokens(4096)
    with pytest.raises(BudgetExhaustedError, match="output_tokens"):
        tracker.record_output_tokens(1)


def test_tracker_reasoning_token_exhaustion_thinking():
    tracker = _make_tracker(THINKING_BUDGET)
    tracker.record_reasoning_tokens(16384)
    with pytest.raises(BudgetExhaustedError, match="reasoning_tokens"):
        tracker.record_reasoning_tokens(1)


def test_tracker_reasoning_disallowed_in_instant():
    tracker = _make_tracker(INSTANT_BUDGET)
    with pytest.raises(BudgetExhaustedError, match="reasoning_tokens"):
        tracker.record_reasoning_tokens(1)


def test_tracker_latency_exhaustion():
    clock = [0.0]
    tracker = _make_tracker(INSTANT_BUDGET, clock_value=clock)
    clock[0] = 14.5
    tracker.check()
    clock[0] = 15.5
    with pytest.raises(BudgetExhaustedError, match="latency"):
        tracker.check()


def test_tracker_latency_exhaustion_thinking():
    clock = [0.0]
    tracker = _make_tracker(THINKING_BUDGET, clock_value=clock)
    clock[0] = 75.0
    with pytest.raises(BudgetExhaustedError, match="latency"):
        tracker.check()


def test_budget_exhausted_error_attributes():
    err = BudgetExhaustedError("output_tokens", 1025, 1024)
    assert err.resource == "output_tokens"
    assert err.used == 1025
    assert err.limit == 1024
    assert "output_tokens" in str(err)


def test_tracker_negative_token_count_rejected():
    tracker = _make_tracker(THINKING_BUDGET)
    with pytest.raises(ValueError, match="non-negative"):
        tracker.record_output_tokens(-1)
    with pytest.raises(ValueError, match="non-negative"):
        tracker.record_reasoning_tokens(-1)


# -- RunBudget / RunCostTracker -----------------------------------------------


def test_run_budget_is_frozen():
    rb = RunBudget(max_usd=30.0)
    assert rb.max_usd == 30.0
    with pytest.raises(AttributeError):
        rb.max_usd = 50.0  # type: ignore[misc]


def test_run_cost_tracker_initial_state():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    assert tracker.used_usd == 0.0
    assert tracker.remaining_usd == 10.0
    assert tracker.llm_cost_usd == 0.0
    assert tracker.tool_cost_usd == 0.0


def test_run_cost_tracker_charge_llm():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    tracker.charge("llm", 2.5)
    assert tracker.llm_cost_usd == pytest.approx(2.5)
    assert tracker.tool_cost_usd == 0.0
    assert tracker.used_usd == pytest.approx(2.5)
    assert tracker.remaining_usd == pytest.approx(7.5)


def test_run_cost_tracker_charge_tool():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    tracker.charge("tool", 1.0)
    assert tracker.tool_cost_usd == pytest.approx(1.0)
    assert tracker.llm_cost_usd == 0.0
    assert tracker.used_usd == pytest.approx(1.0)


def test_run_cost_tracker_mixed_charges():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    tracker.charge("llm", 3.0)
    tracker.charge("tool", 1.5)
    tracker.charge("llm", 2.0)
    assert tracker.llm_cost_usd == pytest.approx(5.0)
    assert tracker.tool_cost_usd == pytest.approx(1.5)
    assert tracker.used_usd == pytest.approx(6.5)
    assert tracker.remaining_usd == pytest.approx(3.5)


def test_run_cost_tracker_charge_unknown_kind():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    with pytest.raises(ValueError, match="unknown cost kind"):
        tracker.charge("storage", 1.0)


def test_run_cost_tracker_charge_negative_rejected():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    with pytest.raises(ValueError, match="non-negative"):
        tracker.charge("llm", -1.0)


def test_run_cost_tracker_check_can_afford_passes():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    tracker.charge("llm", 5.0)
    tracker.check_can_afford(4.99)


def test_run_cost_tracker_check_can_afford_raises():
    tracker = RunCostTracker(budget=RunBudget(max_usd=10.0))
    tracker.charge("llm", 8.0)
    with pytest.raises(RunBudgetExhaustedError) as exc_info:
        tracker.check_can_afford(3.0)
    assert exc_info.value.required_usd == pytest.approx(3.0)
    assert exc_info.value.remaining_usd == pytest.approx(2.0)


def test_run_cost_tracker_remaining_floors_at_zero():
    tracker = RunCostTracker(budget=RunBudget(max_usd=1.0))
    tracker.charge("llm", 5.0)
    assert tracker.remaining_usd == 0.0


def test_run_budget_exhausted_error_message():
    err = RunBudgetExhaustedError(required_usd=5.0, remaining_usd=2.0)
    assert err.required_usd == 5.0
    assert err.remaining_usd == 2.0
    assert "$5.0000" in str(err)
    assert "$2.0000" in str(err)
