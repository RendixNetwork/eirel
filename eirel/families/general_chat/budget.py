from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

ChatMode = Literal["instant", "thinking"]


class BudgetExhaustedError(RuntimeError):
    """Raised when a general_chat budget limit is exceeded."""

    def __init__(self, resource: str, used: int | float, limit: int | float) -> None:
        super().__init__(f"{resource} budget exhausted: {used}/{limit}")
        self.resource = resource
        self.used = used
        self.limit = limit


@dataclass(slots=True, frozen=True)
class ModeBudget:
    """Hard caps for a single general_chat mode + web_search variant."""

    mode: ChatMode
    web_search: bool
    latency_seconds: float
    output_tokens: int
    reasoning_tokens: int


# -- Mode budget constants ----------------------------------------------------

INSTANT_BUDGET = ModeBudget(
    mode="instant",
    web_search=False,
    latency_seconds=15.0,
    output_tokens=1024,
    reasoning_tokens=0,
)

INSTANT_WEB_SEARCH_BUDGET = ModeBudget(
    mode="instant",
    web_search=True,
    latency_seconds=20.0,
    output_tokens=1024,
    reasoning_tokens=0,
)

THINKING_BUDGET = ModeBudget(
    mode="thinking",
    web_search=False,
    latency_seconds=60.0,
    output_tokens=4096,
    reasoning_tokens=16384,
)

THINKING_WEB_SEARCH_BUDGET = ModeBudget(
    mode="thinking",
    web_search=True,
    latency_seconds=75.0,
    output_tokens=4096,
    reasoning_tokens=16384,
)


def get_budget(mode: ChatMode, web_search: bool) -> ModeBudget:
    """Look up the budget for a (mode, web_search) combination."""
    if mode == "instant":
        return INSTANT_WEB_SEARCH_BUDGET if web_search else INSTANT_BUDGET
    if mode == "thinking":
        return THINKING_WEB_SEARCH_BUDGET if web_search else THINKING_BUDGET
    raise ValueError(f"unknown chat mode {mode!r}")


# -- Per-turn tracker ---------------------------------------------------------


@dataclass(slots=True)
class BudgetTracker:
    """Mutable, per-turn usage tracker for a general_chat conversation turn.

    Enforces latency, output_tokens, and reasoning_tokens caps from the
    supplied ``ModeBudget``.
    """

    budget: ModeBudget
    _clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    _start_time: float = field(init=False)
    output_tokens_used: int = field(default=0, init=False)
    reasoning_tokens_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._start_time = self._clock()

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._start_time

    def record_output_tokens(self, n: int) -> None:
        if n < 0:
            raise ValueError("output token count must be non-negative")
        self.output_tokens_used += n
        if self.output_tokens_used > self.budget.output_tokens:
            raise BudgetExhaustedError(
                "output_tokens",
                self.output_tokens_used,
                self.budget.output_tokens,
            )

    def record_reasoning_tokens(self, n: int) -> None:
        if n < 0:
            raise ValueError("reasoning token count must be non-negative")
        self.reasoning_tokens_used += n
        if self.reasoning_tokens_used > self.budget.reasoning_tokens:
            raise BudgetExhaustedError(
                "reasoning_tokens",
                self.reasoning_tokens_used,
                self.budget.reasoning_tokens,
            )

    def check(self) -> None:
        elapsed = self.elapsed_seconds
        if elapsed >= self.budget.latency_seconds:
            raise BudgetExhaustedError(
                "latency",
                round(elapsed, 3),
                self.budget.latency_seconds,
            )
        if self.output_tokens_used > self.budget.output_tokens:
            raise BudgetExhaustedError(
                "output_tokens",
                self.output_tokens_used,
                self.budget.output_tokens,
            )
        if self.reasoning_tokens_used > self.budget.reasoning_tokens:
            raise BudgetExhaustedError(
                "reasoning_tokens",
                self.reasoning_tokens_used,
                self.budget.reasoning_tokens,
            )


# -- Per-run USD budget -------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RunBudget:
    max_usd: float


class RunBudgetExhaustedError(RuntimeError):

    def __init__(self, required_usd: float, remaining_usd: float) -> None:
        super().__init__(
            f"run budget exhausted: need ${required_usd:.4f}, "
            f"only ${remaining_usd:.4f} remaining"
        )
        self.required_usd = required_usd
        self.remaining_usd = remaining_usd


@dataclass(slots=True)
class RunCostTracker:
    budget: RunBudget
    _llm_cost_usd: float = field(default=0.0, init=False)
    _tool_cost_usd: float = field(default=0.0, init=False)

    def charge(self, kind: str, amount_usd: float) -> None:
        if amount_usd < 0:
            raise ValueError("charge amount must be non-negative")
        if kind == "llm":
            self._llm_cost_usd += amount_usd
        elif kind == "tool":
            self._tool_cost_usd += amount_usd
        else:
            raise ValueError(f"unknown cost kind: {kind!r}")

    @property
    def llm_cost_usd(self) -> float:
        return self._llm_cost_usd

    @property
    def tool_cost_usd(self) -> float:
        return self._tool_cost_usd

    @property
    def used_usd(self) -> float:
        return self._llm_cost_usd + self._tool_cost_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget.max_usd - self.used_usd)

    def check_can_afford(self, estimated_usd: float) -> None:
        if self.used_usd + estimated_usd > self.budget.max_usd:
            raise RunBudgetExhaustedError(
                required_usd=estimated_usd,
                remaining_usd=self.remaining_usd,
            )
