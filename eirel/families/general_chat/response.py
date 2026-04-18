from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    url: str = Field(..., min_length=1)
    snippet: str | None = None
    tool_name: str = Field(..., min_length=1)


class ToolCall(BaseModel):
    tool_name: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    result_digest: str | None = None
    latency_ms: int = Field(ge=0, default=0)


class GeneralChatResponse(BaseModel):
    content: str
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    turns: list[GeneralChatResponse] = Field(default_factory=list)


class TraceRecorder:
    """Per-turn recorder; freeze() bakes the captured state into a response."""

    def __init__(self) -> None:
        self._tool_calls: list[ToolCall] = []
        self._citations: list[Citation] = []
        self._content_parts: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._output_tokens: int = 0

    def add_content(self, text: str) -> None:
        if text:
            self._content_parts.append(text)

    def record_output_tokens(self, n: int) -> None:
        if n < 0:
            raise ValueError("output token count must be non-negative")
        self._output_tokens += n

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    def record_tool_call(self, call: ToolCall) -> None:
        self._tool_calls.append(call)

    def record_citation(self, citation: Citation) -> None:
        self._citations.append(citation)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    @property
    def tool_calls(self) -> list[ToolCall]:
        return list(self._tool_calls)

    @property
    def citations(self) -> list[Citation]:
        return list(self._citations)

    def freeze(self) -> GeneralChatResponse:
        return GeneralChatResponse(
            content="".join(self._content_parts),
            citations=list(self._citations),
            tool_calls=list(self._tool_calls),
            metadata=dict(self._metadata),
        )
