from __future__ import annotations

"""Agent-to-Agent (A2A) protocol support.

Implements Google's A2A protocol models and converters for interop with
external A2A-compliant agents.  See: https://a2a-protocol.org
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse


# ── A2A Models ──────────────────────────────────────────────────────────────


class AgentCard(BaseModel):
    """A2A Agent Card — advertises an agent's capabilities."""
    name: str
    description: str = ""
    url: str
    version: str = "1.0"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATaskRequest(BaseModel):
    """A2A tasks/send request body."""
    id: str
    message: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATaskResponse(BaseModel):
    """A2A tasks/send or tasks/get response."""
    id: str
    status: Literal["submitted", "working", "completed", "failed", "canceled"] = "submitted"
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Converters ──────────────────────────────────────────────────────────────


def a2a_request_from_invocation(request: AgentInvocationRequest) -> A2ATaskRequest:
    """Convert an Eirel AgentInvocationRequest to an A2A task request."""
    return A2ATaskRequest(
        id=request.task_id or "",
        message={
            "role": "user",
            "parts": [
                {"type": "text", "text": request.prompt or ""},
            ],
        },
        metadata={
            "family_id": request.family_id,
            "session_id": request.session_id,
            "workflow_spec_id": request.workflow_spec_id,
            "planner_node_id": request.planner_node_id,
            "upstream_node_outputs": request.upstream_node_outputs,
        },
    )


def invocation_response_from_a2a(
    a2a_response: A2ATaskResponse,
    *,
    task_id: str,
    family_id: str,
) -> AgentInvocationResponse:
    """Convert an A2A task response to an Eirel AgentInvocationResponse."""
    status_map = {
        "completed": "completed",
        "failed": "failed",
        "working": "deferred",
        "submitted": "deferred",
        "canceled": "failed",
    }
    eirel_status = status_map.get(a2a_response.status, "failed")

    output = dict(a2a_response.result)
    # Extract text from A2A artifacts
    for artifact in a2a_response.artifacts:
        parts = artifact.get("parts", [])
        for part in parts:
            if part.get("type") == "text" and "text" in part:
                output.setdefault("summary", part["text"])

    return AgentInvocationResponse(
        task_id=task_id,
        family_id=family_id,
        status=eirel_status,
        output=output,
        metadata={
            "a2a_task_id": a2a_response.id,
            "a2a_status": a2a_response.status,
            "a2a_metadata": a2a_response.metadata,
        },
    )
