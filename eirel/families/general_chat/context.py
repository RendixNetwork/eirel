from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from eirel.families.general_chat.budget import ChatMode, ModeBudget, get_budget


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class GeneralChatContext:
    hotkey: str | None
    conversation_id: str
    mode: ChatMode
    web_search_enabled: bool
    budget: ModeBudget
    conversation_history: tuple[ConversationTurn, ...]


def context_from_request(request_payload: dict[str, Any]) -> GeneralChatContext:
    """Construct a GeneralChatContext from a raw invocation payload.

    Reads the slim 0.3.0 contract first (top-level ``mode`` /
    ``web_search`` / ``history``) and falls back to the legacy 0.2.x
    shape (``inputs.mode`` / ``inputs.web_search`` / ``context_history``)
    for one release so older orchestrators / validators keep working.
    """
    inputs: dict[str, Any] = request_payload.get("inputs") or {}

    # Mode: top-level wins, then inputs.mode, default instant.
    raw_mode = str(
        request_payload.get("mode")
        or inputs.get("mode")
        or "instant"
    )
    if raw_mode not in ("instant", "thinking"):
        raise ValueError(f"unsupported mode {raw_mode!r}; expected 'instant' or 'thinking'")
    mode: ChatMode = raw_mode  # type: ignore[assignment]

    # web_search: top-level wins, then inputs.web_search, default false.
    if "web_search" in request_payload and isinstance(request_payload["web_search"], bool):
        web_search = bool(request_payload["web_search"])
    else:
        web_search = bool(inputs.get("web_search", False))
    budget = get_budget(mode, web_search)

    conversation_id = str(
        request_payload.get("turn_id")
        or request_payload.get("conversation_id")
        or request_payload.get("session_id")
        or request_payload.get("task_id")
        or ""
    )
    hotkey = request_payload.get("hotkey") or request_payload.get("miner_hotkey")
    hotkey = str(hotkey) if hotkey else None

    # History: prefer the slim ``history`` field, fall back to legacy
    # ``context_history`` / ``inputs.conversation_history`` / ``messages``.
    raw_history: list[Any] = list(
        request_payload.get("history")
        or request_payload.get("context_history")
        or inputs.get("conversation_history")
        or request_payload.get("conversation_history")
        or request_payload.get("messages")
        or []
    )
    history: list[ConversationTurn] = []
    for entry in raw_history:
        if isinstance(entry, ConversationTurn):
            history.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "user")
        if role not in ("user", "assistant"):
            continue
        history.append(
            ConversationTurn(
                role=role,  # type: ignore[arg-type]
                content=str(entry.get("content") or ""),
                metadata=entry.get("metadata"),
            )
        )

    return GeneralChatContext(
        hotkey=hotkey,
        conversation_id=conversation_id,
        mode=mode,
        web_search_enabled=web_search,
        budget=budget,
        conversation_history=tuple(history),
    )


__all__ = [
    "ConversationTurn",
    "GeneralChatContext",
    "context_from_request",
]
