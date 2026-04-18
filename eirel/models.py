from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Evaluated at module import time. Tests that need to flip these should
# monkeypatch the module attribute directly.
MAX_MESSAGES = _env_int("EIREL_MAX_MESSAGES", 100)
MAX_MESSAGE_CONTENT_BYTES = _env_int("EIREL_MAX_MESSAGE_CONTENT_BYTES", 32_768)


class ToolFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: ToolFunctionDefinition


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @field_validator("content")
    @classmethod
    def enforce_content_size(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value.encode("utf-8")) > MAX_MESSAGE_CONTENT_BYTES:
            raise ValueError(
                f"message content exceeds EIREL_MAX_MESSAGE_CONTENT_BYTES "
                f"({MAX_MESSAGE_CONTENT_BYTES} bytes)"
            )
        return value

    @model_validator(mode="after")
    def validate_message_shape(self):
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages must include tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are only allowed on assistant messages")
        if self.content is None and not self.tool_calls:
            raise ValueError("message content may only be null when tool_calls are present")
        return self


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int | None = None

    @field_validator("messages")
    @classmethod
    def enforce_message_count(cls, value: list[Message]) -> list[Message]:
        if len(value) > MAX_MESSAGES:
            raise ValueError(
                f"messages list exceeds EIREL_MAX_MESSAGES ({MAX_MESSAGES})"
            )
        return value

    @field_validator("temperature")
    @classmethod
    def ensure_temperature_is_number(cls, value: float) -> float:
        return float(value)


class ChatCompletionResponse(BaseModel):
    choices: list[Choice]

    @model_validator(mode="after")
    def validate_choices(self):
        if not self.choices:
            raise ValueError("choices must contain at least one entry")
        return self
