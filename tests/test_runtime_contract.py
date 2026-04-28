from __future__ import annotations

from eirel.helpers import (
    build_tool_call,
    chat_payload_from_agent_request,
    infer_response_from_chat_payload,
    workflow_completed_response,
    workflow_deferred_response,
    workflow_failed_response,
    workflow_request_context,
)
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse


def test_agent_invocation_request_accepts_v3_runtime_fields():
    request = AgentInvocationRequest.model_validate(
        {
            "task_id": "task-1",
            "session_id": "session-1",
            "primary_goal": "Research, build, and verify.",
            "subtask": "Produce the next step.",
            "family_id": "general_chat",
            "episode_id": "episode-1",
            "workflow_spec_id": "research_build_verify_v1",
            "workflow_version": "v1",
            "planner_node_id": "analyst_plan",
            "role_id": "research_plan",
            "upstream_node_outputs": {"control_plane_context_pack": {"output": {"summary": "ctx"}}},
            "context_bundle": {"workflow_class": "research_build_verify"},
            "checkpoint_state": {"draft": "step-1"},
            "resume_token": "resume-1",
            "artifact_requirements": {"report": "json"},
            "trace_policy": {"capture_tool_calls": True},
        }
    )

    assert request.episode_id == "episode-1"
    assert request.workflow_spec_id == "research_build_verify_v1"
    assert request.planner_node_id == "analyst_plan"
    assert request.context_bundle == {"workflow_class": "research_build_verify"}
    assert request.checkpoint_state == {"draft": "step-1"}


def test_agent_invocation_response_accepts_v3_runtime_fields():
    # 0.3.0 contract: top-level ``tool_calls`` is gone; executed tool
    # calls live under ``metadata.executed_tool_calls``.
    response = AgentInvocationResponse.model_validate(
        {
            "task_id": "task-1",
            "family_id": "general_chat",
            "status": "deferred",
            "output": {"code": "draft"},
            "checkpoint_events": [{"event": "paused", "checkpoint_id": "cp-1"}],
            "runtime_state_patch": {"draft": "step-1"},
            "resume_token": "resume-1",
            "metadata": {"executed_tool_calls": [{"name": "lookup_spec"}]},
            "reliability_score": 0.91,
            "recovery_score": 0.44,
        }
    )

    assert response.checkpoint_events == [{"event": "paused", "checkpoint_id": "cp-1"}]
    assert response.runtime_state_patch == {"draft": "step-1"}
    assert response.resume_token == "resume-1"
    assert response.metadata["executed_tool_calls"] == [{"name": "lookup_spec"}]
    assert response.reliability_score == 0.91
    assert response.recovery_score == 0.44


def test_legacy_request_payload_still_validates_without_v3_runtime_fields():
    request = AgentInvocationRequest.model_validate(
        {
            "task_id": "task-legacy",
            "primary_goal": "Answer the question.",
            "subtask": "Respond directly.",
            "family_id": "general_chat",
        }
    )

    assert request.workflow_spec_id is None
    assert request.checkpoint_state == {}
    assert request.trace_policy == {}


def test_workflow_request_context_prefers_canonical_fields_then_legacy_fallbacks():
    canonical = AgentInvocationRequest.model_validate(
        {
            "task_id": "task-canonical",
            "primary_goal": "Do workflow work.",
            "subtask": "Step",
            "family_id": "general_chat",
            "workflow_spec_id": "analysis_verify_v1",
            "planner_node_id": "analyst_plan",
            "role_id": "analysis",
            "context_bundle": {"mode": "canonical"},
            "checkpoint_state": {"draft": "v3"},
            "trace_policy": {"capture_tool_calls": True},
            "inputs": {
                "context_bundle": {"mode": "legacy"},
                "checkpoint_state": {"draft": "legacy"},
            },
            "metadata": {
                "planner_node_id": "legacy-node",
                "trace_policy": {"capture_tool_calls": False},
            },
        }
    )
    legacy = AgentInvocationRequest.model_validate(
        {
            "task_id": "task-legacy",
            "primary_goal": "Do workflow work.",
            "subtask": "Step",
            "family_id": "general_chat",
            "inputs": {
                "workflow_episode_id": "episode-legacy",
                "upstream_node_outputs": {"analyst_plan": {"output": {"summary": "plan"}}},
                "context_bundle": {"mode": "legacy"},
                "checkpoint_state": {"draft": "legacy"},
                "resume_token": "resume-legacy",
                "artifact_requirements": {"code": "python"},
            },
            "metadata": {
                "workflow_spec_id": "research_build_verify_v1",
                "workflow_version": "v1",
                "planner_node_id": "builder_impl",
                "role_id": "implementation",
                "trace_policy": {"capture_state_patch": True},
            },
        }
    )

    canonical_context = workflow_request_context(canonical)
    legacy_context = workflow_request_context(legacy)

    assert canonical_context["planner_node_id"] == "analyst_plan"
    assert canonical_context["context_bundle"] == {"mode": "canonical"}
    assert canonical_context["checkpoint_state"] == {"draft": "v3"}
    assert legacy_context["episode_id"] == "episode-legacy"
    assert legacy_context["workflow_spec_id"] == "research_build_verify_v1"
    assert legacy_context["planner_node_id"] == "builder_impl"
    assert legacy_context["artifact_requirements"] == {"code": "python"}


def test_workflow_response_helpers_emit_canonical_and_legacy_runtime_fields():
    completed = workflow_completed_response(
        task_id="task-1",
        family_id="general_chat",
        output={"summary": "done"},
        checkpoint_events=[{"event": "completed"}],
        runtime_state_patch={"draft": "done"},
        reliability_score=0.9,
        metadata={"handled": True},
    )
    deferred = workflow_deferred_response(
        task_id="task-1",
        family_id="general_chat",
        output={"code": "partial"},
        resume_token="resume-1",
        checkpoint_events=[{"event": "paused"}],
        runtime_state_patch={"draft": "step-1"},
    )
    failed = workflow_failed_response(
        task_id="task-1",
        family_id="general_chat",
        error="bad artifact",
        checkpoint_events=[],
        runtime_state_patch={},
    )

    assert completed.checkpoint_events == [{"event": "completed"}]
    assert completed.metadata["runtime_state_patch"] == {"draft": "done"}
    assert completed.metadata["reliability_score"] == 0.9
    assert deferred.status == "deferred"
    assert deferred.resume_token == "resume-1"
    assert deferred.metadata["resume_token"] == "resume-1"
    assert failed.status == "failed"
    assert failed.error == "bad artifact"


def test_chat_helpers_preserve_runtime_context_and_tool_calls():
    request = AgentInvocationRequest(
        task_id="task-1",
        primary_goal="Research the problem.",
        subtask="Write a plan.",
        family_id="general_chat",
        episode_id="episode-1",
        workflow_spec_id="analysis_verify_v1",
        workflow_version="v1",
        planner_node_id="analyst_plan",
        role_id="analysis",
        checkpoint_state={"draft": "step-1"},
        resume_token="resume-1",
    )

    chat_payload = chat_payload_from_agent_request(request)
    assert chat_payload["agent_invocation"]["workflow_spec_id"] == "analysis_verify_v1"
    assert chat_payload["agent_invocation"]["planner_node_id"] == "analyst_plan"

    agent_response = infer_response_from_chat_payload(
        request=request,
        payload={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "Plan complete.",
                        "tool_calls": [
                            build_tool_call(
                                tool_id="call-1",
                                name="retrieval_search",
                                arguments={"query": "runtime contract"},
                            ).model_dump(mode="json")
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )

    assert (
        agent_response["metadata"]["executed_tool_calls"][0]["function"]["name"]
        == "retrieval_search"
    )
    assert agent_response["metadata"]["compatibility_mode"] == "chat_completions"

