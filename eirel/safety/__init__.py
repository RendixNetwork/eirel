"""Pluggable safety guards for graph runs."""
from __future__ import annotations

from eirel.safety.chain import ChainedGuard
from eirel.safety.guard import Guard, GuardVerdict
from eirel.safety.noop import NoopGuard


def __getattr__(name: str):
    if name == "LlamaGuardAdapter":
        from eirel.safety.llama_guard import LlamaGuardAdapter  # type: ignore[import-not-found]
        return LlamaGuardAdapter
    if name == "NeMoGuardAdapter":
        from eirel.safety.nemo import NeMoGuardAdapter  # type: ignore[import-not-found]
        return NeMoGuardAdapter
    raise AttributeError(f"module 'eirel.safety' has no attribute {name!r}")


__all__ = [
    "ChainedGuard",
    "Guard",
    "GuardVerdict",
    "LlamaGuardAdapter",
    "NeMoGuardAdapter",
    "NoopGuard",
]
