"""Compose multiple guards into one.

``ChainedGuard([a, b, c])`` runs ``a`` first; if it denies, the chain
short-circuits and returns the deny verdict. If it allows, ``b`` runs,
and so on. ``redactions`` from earlier guards merge into the state seen
by later guards (callers wire that up via the runtime's standard
reducer path).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eirel.safety.guard import Guard, GuardVerdict

__all__ = ["ChainedGuard"]


class ChainedGuard(Guard):
    """Run a sequence of guards in order; first deny wins."""

    def __init__(self, guards: Sequence[Guard]) -> None:
        if not guards:
            raise ValueError("ChainedGuard requires at least one guard")
        self._guards = list(guards)

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)

    async def pre_input(self, state: Mapping[str, Any]) -> GuardVerdict:
        accumulated_meta: dict[str, Any] = {}
        merged_redactions: dict[str, Any] | None = None
        for idx, guard in enumerate(self._guards):
            verdict = await guard.pre_input(state)
            accumulated_meta[f"guard_{idx}"] = dict(verdict.metadata or {})
            if not verdict.allow:
                return GuardVerdict(
                    allow=False,
                    reason=verdict.reason or f"guard {idx} denied",
                    redactions=verdict.redactions,
                    metadata={
                        "denied_by_index": idx,
                        "chain": accumulated_meta,
                    },
                )
            if verdict.redactions:
                merged_redactions = {
                    **(merged_redactions or {}),
                    **dict(verdict.redactions),
                }
        return GuardVerdict(
            allow=True, redactions=merged_redactions, metadata={"chain": accumulated_meta}
        )

    async def post_output(self, state: Mapping[str, Any]) -> GuardVerdict:
        accumulated_meta: dict[str, Any] = {}
        merged_redactions: dict[str, Any] | None = None
        for idx, guard in enumerate(self._guards):
            verdict = await guard.post_output(state)
            accumulated_meta[f"guard_{idx}"] = dict(verdict.metadata or {})
            if not verdict.allow:
                return GuardVerdict(
                    allow=False,
                    reason=verdict.reason or f"guard {idx} denied",
                    redactions=verdict.redactions,
                    metadata={
                        "denied_by_index": idx,
                        "chain": accumulated_meta,
                    },
                )
            if verdict.redactions:
                merged_redactions = {
                    **(merged_redactions or {}),
                    **dict(verdict.redactions),
                }
        return GuardVerdict(
            allow=True, redactions=merged_redactions, metadata={"chain": accumulated_meta}
        )

    async def aclose(self) -> None:
        for guard in self._guards:
            try:
                await guard.aclose()
            except Exception:  # noqa: BLE001
                pass
