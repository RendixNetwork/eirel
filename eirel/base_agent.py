from __future__ import annotations

from abc import ABC, abstractmethod

from eirel.schemas import (
    AgentCapabilityMetadata,
    AgentHealthStatus,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentRegistrationMetadata,
)


class BaseAgent(ABC):
    def __init__(self, *, hotkey: str, endpoint: str, version: str, capabilities: AgentCapabilityMetadata):
        self.hotkey = hotkey
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.capabilities = capabilities

    @abstractmethod
    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        raise NotImplementedError

    async def health(self) -> AgentHealthStatus:
        return AgentHealthStatus(
            status="ok",
            family_id=self.capabilities.family_id,
            version=self.version,
            details={"endpoint": self.endpoint},
        )

    def registration(self) -> AgentRegistrationMetadata:
        return AgentRegistrationMetadata(
            hotkey=self.hotkey,
            endpoint=self.endpoint,
            family_id=self.capabilities.family_id,
            version=self.version,
            capabilities=self.capabilities,
        )
