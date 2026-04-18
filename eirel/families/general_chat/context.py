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

    The payload should at minimum carry ``conversation_id`` (or
    ``task_id``), ``inputs.mode``, ``inputs.web_search``, and either an
    explicit ``inputs.conversation_history`` list or a ``messages`` list in
    chat-completions format.
    """
    inputs: dict[str, Any] = request_payload.get("inputs") or {}
    raw_mode = str(inputs.get("mode") or "instant")
    if raw_mode not in ("instant", "thinking"):
        raise ValueError(f"unsupported mode {raw_mode!r}; expected 'instant' or 'thinking'")
    mode: ChatMode = raw_mode  # type: ignore[assignment]

    web_search = bool(inputs.get("web_search", False))
    budget = get_budget(mode, web_search)

    conversation_id = str(
        request_payload.get("conversation_id")
        or request_payload.get("session_id")
        or request_payload.get("task_id")
        or ""
    )
    hotkey = request_payload.get("hotkey") or request_payload.get("miner_hotkey")
    hotkey = str(hotkey) if hotkey else None

    raw_history: list[Any] = list(
        inputs.get("conversation_history")
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
