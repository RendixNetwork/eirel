from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class XApiTool(GeneralChatTool):
    """Owner-api-routed X (formerly Twitter) search.

    Note: enforced by owner-api at **1 call per task maximum**. The SDK does
    not enforce the cap locally; it only reports the call to the budget
    tracker for accounting.
    """

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "x_api"

    @property
    def description(self) -> str:
        return (
            "Search X (formerly Twitter) for recent posts via the owner-hosted "
            "X tool service. Hard cap: 1 call per task."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of posts to return.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs["query"])
        max_results = int(kwargs.get("max_results", 10))
        max_results = max(1, min(max_results, 50))
        return await self._client.request(
            path="/v1/tools/x_api",
            payload={"query": query, "max_results": max_results},
        )


__all__ = ["XApiTool"]
