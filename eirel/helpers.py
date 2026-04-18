from __future__ import annotations

import json
import re
from typing import Any

from eirel.models import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ToolCall
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse, ContextMessage
from eirel.token_signing import sign_resume_token as _sign_resume_token


def _maybe_sign_token(token: str | None, secret: str | None) -> str | None:
    """Sign a resume token if both token and secret are provided."""
    if token and secret:
        return _sign_resume_token(token, secret)
    return token


def validate_request(payload: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(payload)


def validate_agent_request(payload: dict) -> AgentInvocationRequest:
    return AgentInvocationRequest.model_validate(payload)


def _request_field_present(request: AgentInvocationRequest, field_name: str) -> bool:
    return field_name in request.model_fields_set


def _request_runtime_value(
    request: AgentInvocationRequest,
    *,
    field_name: str,
    legacy_input_key: str | None = None,
    legacy_metadata_key: str | None = None,
    default: Any = None,
) -> Any:
    value = getattr(request, field_name)
    if _request_field_present(request, field_name):
        return value
    if value not in (None, {}, [], ""):
        return value
    if legacy_input_key is not None:
        legacy_input = request.inputs.get(legacy_input_key)
        if legacy_input not in (None, {}, [], ""):
            return legacy_input
    if legacy_metadata_key is not None:
        legacy_metadata = request.metadata.get(legacy_metadata_key)
        if legacy_metadata not in (None, {}, [], ""):
            return legacy_metadata
    return default


def content_response(content: str, *, finish_reason: str = "stop") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        choices=[
            Choice(
                index=0,
                message=AssistantMessage(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def tool_call_response(
    tool_calls: list[ToolCall],
    *,
    finish_reason: str = "tool_calls",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        choices=[
            Choice(
                index=0,
                message=AssistantMessage(content=None, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def build_tool_call(*, tool_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        }
    )


def workflow_request_context(request: AgentInvocationRequest) -> dict[str, Any]:
    return {
        "episode_id": _request_runtime_value(
            request,
            field_name="episode_id",
            legacy_input_key="workflow_episode_id",
            default=request.session_id,
        ),
        "workflow_spec_id": _request_runtime_value(
            request,
            field_name="workflow_spec_id",
            legacy_metadata_key="workflow_spec_id",
        ),
        "workflow_version": _request_runtime_value(
            request,
            field_name="workflow_version",
            legacy_metadata_key="workflow_version",
        ),
        "planner_node_id": _request_runtime_value(
            request,
            field_name="planner_node_id",
            legacy_metadata_key="planner_node_id",
        ),
        "role_id": _request_runtime_value(
            request,
            field_name="role_id",
            legacy_metadata_key="role_id",
        ),
        "upstream_node_outputs": _request_runtime_value(
            request,
            field_name="upstream_node_outputs",
            legacy_input_key="upstream_node_outputs",
            default={},
        ),
        "context_bundle": _request_runtime_value(
            request,
            field_name="context_bundle",
            legacy_input_key="context_bundle",
            default={},
        ),
        "checkpoint_state": _request_runtime_value(
            request,
            field_name="checkpoint_state",
            legacy_input_key="checkpoint_state",
            default={},
        ),
        "resume_token": _request_runtime_value(
            request,
            field_name="resume_token",
            legacy_input_key="resume_token",
        ),
        "artifact_requirements": _request_runtime_value(
            request,
            field_name="artifact_requirements",
            legacy_input_key="artifact_requirements",
            default={},
        ),
        "trace_policy": _request_runtime_value(
            request,
            field_name="trace_policy",
            legacy_metadata_key="trace_policy",
            default={},
        ),
    }


def workflow_completed_response(
    *,
    task_id: str,
    family_id: str,
    output: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
    citations: list[str] | None = None,
    latency_ms: int = 0,
    cost_tao: float = 0.0,
    checkpoint_events: list[dict[str, Any]] | None = None,
    runtime_state_patch: dict[str, Any] | None = None,
    resume_token: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    reliability_score: float | None = None,
    recovery_score: float | None = None,
    metadata: dict[str, Any] | None = None,
    signing_secret: str | None = None,
) -> AgentInvocationResponse:
    resume_token = _maybe_sign_token(resume_token, signing_secret)
    merged_metadata = dict(metadata or {})
    merged_metadata["checkpoint_events"] = list(checkpoint_events or [])
    merged_metadata["runtime_state_patch"] = dict(runtime_state_patch or {})
    if resume_token:
        merged_metadata["resume_token"] = resume_token
    if tool_calls:
        merged_metadata["tool_calls"] = list(tool_calls)
    if reliability_score is not None:
        merged_metadata["reliability_score"] = reliability_score
    if recovery_score is not None:
        merged_metadata["recovery_score"] = recovery_score
    return AgentInvocationResponse(
        task_id=task_id,
        family_id=family_id,
        status="completed",
        output=dict(output or {}),
        artifacts=list(artifacts or []),
        citations=list(citations or []),
        latency_ms=latency_ms,
        cost_tao=cost_tao,
        checkpoint_events=list(checkpoint_events or []),
        runtime_state_patch=dict(runtime_state_patch or {}),
        resume_token=resume_token,
        tool_calls=list(tool_calls or []),
        reliability_score=reliability_score,
        recovery_score=recovery_score,
        metadata=merged_metadata,
    )


def workflow_deferred_response(
    *,
    task_id: str,
    family_id: str,
    output: dict[str, Any],
    resume_token: str,
    checkpoint_events: list[dict[str, Any]],
    runtime_state_patch: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
    citations: list[str] | None = None,
    latency_ms: int = 0,
    cost_tao: float = 0.0,
    tool_calls: list[dict[str, Any]] | None = None,
    reliability_score: float | None = None,
    recovery_score: float | None = None,
    metadata: dict[str, Any] | None = None,
    signing_secret: str | None = None,
) -> AgentInvocationResponse:
    resume_token = _maybe_sign_token(resume_token, signing_secret) or resume_token
    merged_metadata = dict(metadata or {})
    merged_metadata["checkpoint_events"] = list(checkpoint_events or [])
    merged_metadata["runtime_state_patch"] = dict(runtime_state_patch or {})
    merged_metadata["resume_token"] = resume_token
    if tool_calls:
        merged_metadata["tool_calls"] = list(tool_calls)
    if reliability_score is not None:
        merged_metadata["reliability_score"] = reliability_score
    if recovery_score is not None:
        merged_metadata["recovery_score"] = recovery_score
    return AgentInvocationResponse(
        task_id=task_id,
        family_id=family_id,
        status="deferred",
        output=dict(output or {}),
        artifacts=list(artifacts or []),
        citations=list(citations or []),
        latency_ms=latency_ms,
        cost_tao=cost_tao,
        checkpoint_events=list(checkpoint_events or []),
        runtime_state_patch=dict(runtime_state_patch or {}),
        resume_token=resume_token,
        tool_calls=list(tool_calls or []),
        reliability_score=reliability_score,
        recovery_score=recovery_score,
        metadata=merged_metadata,
    )


def workflow_failed_response(
    *,
    task_id: str,
    family_id: str,
    error: str,
    output: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    citations: list[str] | None = None,
    latency_ms: int = 0,
    cost_tao: float = 0.0,
    checkpoint_events: list[dict[str, Any]] | None = None,
    runtime_state_patch: dict[str, Any] | None = None,
    resume_token: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    reliability_score: float | None = None,
    recovery_score: float | None = None,
    metadata: dict[str, Any] | None = None,
    signing_secret: str | None = None,
) -> AgentInvocationResponse:
    resume_token = _maybe_sign_token(resume_token, signing_secret)
    merged_metadata = dict(metadata or {})
    merged_metadata["checkpoint_events"] = list(checkpoint_events or [])
    merged_metadata["runtime_state_patch"] = dict(runtime_state_patch or {})
    if resume_token:
        merged_metadata["resume_token"] = resume_token
    if tool_calls:
        merged_metadata["tool_calls"] = list(tool_calls)
    if reliability_score is not None:
        merged_metadata["reliability_score"] = reliability_score
    if recovery_score is not None:
        merged_metadata["recovery_score"] = recovery_score
    return AgentInvocationResponse(
        task_id=task_id,
        family_id=family_id,
        status="failed",
        output=dict(output or {}),
        artifacts=list(artifacts or []),
        citations=list(citations or []),
        latency_ms=latency_ms,
        cost_tao=cost_tao,
        error=error,
        checkpoint_events=list(checkpoint_events or []),
        runtime_state_patch=dict(runtime_state_patch or {}),
        resume_token=resume_token,
        tool_calls=list(tool_calls or []),
        reliability_score=reliability_score,
        recovery_score=recovery_score,
        metadata=merged_metadata,
    )


def chat_payload_from_agent_request(request: AgentInvocationRequest) -> dict:
    messages: list[dict[str, str]] = []
    if request.primary_goal:
        messages.append({"role": "system", "content": request.primary_goal})
    for item in request.context_history:
        if isinstance(item, ContextMessage):
            messages.append({"role": item.role, "content": item.content})
        else:
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": request.subtask})
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 800,
        "agent_invocation": request.model_dump(mode="json"),
    }
    if request.tools is not None:
        payload["tools"] = [t.model_dump(mode="json") for t in request.tools]
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    return payload


def infer_response_from_chat_payload(
    *,
    request: AgentInvocationRequest,
    payload: dict,
) -> dict:
    response = ChatCompletionResponse.model_validate(payload)
    content = response.choices[0].message.content or ""
    tool_calls = [
        item.model_dump(mode="json")
        for item in (response.choices[0].message.tool_calls or [])
    ]
    output_key = {
        "general_chat": "content",
    }.get(str(request.family_id), "content")
    return AgentInvocationResponse(
        task_id=request.task_id,
        family_id=request.family_id,
        output={output_key: content},
        citations=_extract_citations(content),
        tool_calls=tool_calls,
        metadata={"compatibility_mode": "chat_completions"},
    ).model_dump(mode="json")


def _extract_citations(content: str) -> list[str]:
    return [match.rstrip(".,);]") for match in re.findall(r"https?://[^\s)\]]+", content)]
