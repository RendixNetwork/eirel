"""No-op guard. The runtime's default when none is configured."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eirel.safety.guard import Guard, GuardVerdict

__all__ = ["NoopGuard"]


class NoopGuard(Guard):
    async def pre_input(self, state: Mapping[str, Any]) -> GuardVerdict:  # noqa: ARG002
        return GuardVerdict.ok()

    async def post_output(self, state: Mapping[str, Any]) -> GuardVerdict:  # noqa: ARG002
        return GuardVerdict.ok()
