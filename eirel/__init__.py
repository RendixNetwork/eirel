from eirel.app import MinerApp
from eirel.agent_server import build_agent_app
from eirel.base_agent import BaseAgent
from eirel.families.general_chat import (
    INSTANT_BUDGET,
    INSTANT_WEB_SEARCH_BUDGET,
    THINKING_BUDGET,
    THINKING_WEB_SEARCH_BUDGET,
    BudgetExhaustedError,
    BudgetTracker,
    ChatMode,
    Citation,
    ConversationResponse,
    ConversationTurn,
    GeneralChatContext,
    GeneralChatResponse,
    GeneralChatTool,
    GeneralChatToolCatalog,
    ModeBudget,
    SandboxTool,
    SemanticScholarTool,
    ToolCall,
    TraceRecorder,
    WebSearchTool,
    XApiTool,
    context_from_request,
    get_budget,
)
from eirel.groups import FAMILY_DESCRIPTIONS, FAMILY_IDS, FamilyId
from eirel.helpers import (
    build_tool_call,
    content_response,
    tool_call_response,
    validate_request,
    workflow_completed_response,
    workflow_deferred_response,
    workflow_failed_response,
    workflow_request_context,
)
from eirel.models import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ToolDefinition,
)
from eirel.models import ToolCall as ChatToolCall
from eirel.provider import AgentProviderClient, MinerProviderConfig
from eirel.registration import RegistrationRequest, build_registration_request
from eirel.schemas import (
    AgentCapabilityMetadata,
    AgentHealthStatus,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentRegistrationMetadata,
    ContextMessage,
    InvocationConstraints,
)


# Lazy imports — only available with `pip install eirel[submit]`
def __getattr__(name: str):
    if name in ("Signer", "load_signer"):
        from eirel import signing
        return getattr(signing, name)
    raise AttributeError(f"module 'eirel' has no attribute {name!r}")


__all__ = [
    "AgentCapabilityMetadata",
    "AgentHealthStatus",
    "AgentInvocationRequest",
    "AgentInvocationResponse",
    "AgentProviderClient",
    "AgentRegistrationMetadata",
    "AssistantMessage",
    "BaseAgent",
    "BudgetExhaustedError",
    "BudgetTracker",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMode",
    "ChatToolCall",
    "Citation",
    "ContextMessage",
    "ConversationResponse",
    "ConversationTurn",
    "FAMILY_DESCRIPTIONS",
    "FAMILY_IDS",
    "FamilyId",
    "GeneralChatContext",
    "GeneralChatResponse",
    "GeneralChatTool",
    "GeneralChatToolCatalog",
    "INSTANT_BUDGET",
    "INSTANT_WEB_SEARCH_BUDGET",
    "InvocationConstraints",
    "MinerApp",
    "MinerProviderConfig",
    "ModeBudget",
    "RegistrationRequest",
    "SandboxTool",
    "SemanticScholarTool",
    "Signer",
    "THINKING_BUDGET",
    "THINKING_WEB_SEARCH_BUDGET",
    "ToolCall",
    "ToolDefinition",
    "TraceRecorder",
    "WebSearchTool",
    "XApiTool",
    "build_agent_app",
    "build_registration_request",
    "build_tool_call",
    "content_response",
    "context_from_request",
    "get_budget",
    "load_signer",
    "tool_call_response",
    "validate_request",
    "workflow_completed_response",
    "workflow_deferred_response",
    "workflow_failed_response",
    "workflow_request_context",
]
