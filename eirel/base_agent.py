from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from eirel.schemas import (
    AgentCapabilityMetadata,
    AgentHealthStatus,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentRegistrationMetadata,
    StreamChunk,
)


class BaseAgent(ABC):
    """Abstract base for miner agents.

    The canonical authoring shape is to compose a graph with
    :class:`~eirel.graph.graph.StateGraph`, compile it, and wrap the
    compiled graph in :class:`~eirel.agents.graph_agent.GraphAgent`.
    ``BaseAgent`` is the structural base ``GraphAgent`` inherits from —
    direct subclassing is supported but uncommon.
    """

    def __init__(self, *, hotkey: str, endpoint: str, version: str, capabilities: AgentCapabilityMetadata):
        self.hotkey = hotkey
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.capabilities = capabilities

    @abstractmethod
    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        raise NotImplementedError

    async def infer_stream(
        self, request: AgentInvocationRequest,
    ) -> AsyncIterator[StreamChunk]:
        """Stream the agent's response as NDJSON chunks.

        Default implementation: call non-streaming `infer()` and emit a
        single `delta` carrying the whole answer text, followed by a `done`
        chunk. Existing agents that only override `infer()` keep working —
        the validator will still see them as "streaming-capable" but their
        first-token latency will equal their full completion latency.

        Real streaming agents should override this method to yield `delta`
        chunks as soon as the underlying LLM emits tokens. That's the only
        way to satisfy the 10s TTFB SLA.
        """
        response = await self.infer(request)
        # Best-effort extraction of the answer text from the non-streaming
        # output. Different families nest the answer under different keys;
        # try the common ones in priority order.
        out = response.output or {}
        text = ""
        for key in ("answer", "response", "text", "content", "message"):
            value = out.get(key)
            if isinstance(value, str) and value:
                text = value
                break
        if text:
            yield StreamChunk(event="delta", text=text)
        # Executed tool calls travel inside metadata; pass it through
        # verbatim so anything the agent recorded survives the
        # unary→stream adapter.
        yield StreamChunk(
            event="done",
            output=out,
            citations=list(response.citations or []),
            status=response.status,
            error=response.error,
            metadata=dict(response.metadata or {}),
        )

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
