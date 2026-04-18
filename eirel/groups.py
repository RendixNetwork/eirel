from __future__ import annotations

import os
from typing import Literal, get_args


# Future families: "deep_research", "coding"
FamilyId = Literal["general_chat"]

FAMILY_IDS: tuple[str, ...] = get_args(FamilyId)

# Families available at subnet launch.  When EIREL_LAUNCH_MODE is enabled,
# only these families accept registrations and receive emissions.
LAUNCH_FAMILIES: tuple[str, ...] = ("general_chat",)

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "general_chat": (
        "Multi-turn conversational assistant with optional web search, "
        "across instant and thinking modes. Backed by owner-routed tool "
        "services for web search, X, Semantic Scholar, and a server-side "
        "Python sandbox for verifiable computation."
    ),
}


def is_launch_mode() -> bool:
    """Return True when the subnet is running in launch mode."""
    return os.getenv("EIREL_LAUNCH_MODE", "").lower() in ("1", "true", "yes")


def ensure_family_id(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized not in FAMILY_IDS:
        allowed = ", ".join(FAMILY_IDS)
        raise ValueError(f"unsupported family_id {value!r}; expected one of {allowed}")
    return normalized


def ensure_active_family_id(value: str) -> str:
    """Validate that a family_id is active.

    In launch mode, only LAUNCH_FAMILIES are accepted.
    Otherwise, all FAMILY_IDS are accepted.
    """
    normalized = ensure_family_id(value)
    if is_launch_mode() and normalized not in LAUNCH_FAMILIES:
        allowed = ", ".join(LAUNCH_FAMILIES)
        raise ValueError(
            f"family_id {value!r} is not active in launch mode; "
            f"expected one of {allowed}"
        )
    return normalized
