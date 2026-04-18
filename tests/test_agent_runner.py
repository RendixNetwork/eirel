from __future__ import annotations

from eirel import (
    AgentCapabilityMetadata,
    AgentInvocationRequest,
    AgentInvocationResponse,
    BaseAgent,
    build_agent_app,
    build_registration_request,
)


class DummyResearchAgent(BaseAgent):
    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        return AgentInvocationResponse(
            task_id=request.task_id,
            family_id=request.family_id,
            output={"answer": request.subtask},
            metadata={"handled": True},
        )


def test_agent_app_exposes_registration_and_health():
    agent = DummyResearchAgent(
        hotkey="miner-hotkey",
        endpoint="http://127.0.0.1:9000",
        version="1.0.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat",
            description="Analyst family worker",
            latency_ms_p50=1200,
        ),
    )

    app = build_agent_app(agent)

    registration = agent.registration()
    assert registration.family_id == "general_chat"
    assert registration.endpoint == "http://127.0.0.1:9000"
    assert app.title == "general_chat-agent"


def test_registration_request_uses_group_contract():
    request = build_registration_request(
        hotkey="miner-hotkey",
        endpoint="http://127.0.0.1:9000",
        family_id="general_chat",
    )
    assert request.family_id == "general_chat"
    assert request.cooldown_epochs == 3
