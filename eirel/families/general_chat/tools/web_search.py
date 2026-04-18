from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class WebSearchTool(GeneralChatTool):
    """Owner-api-routed web search."""

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the public web for relevant pages via the owner-hosted "
            "web search service."
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
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs["query"])
        max_results = int(kwargs.get("max_results", 5))
        top_k = max(1, min(max_results, 20))
        return await self._client.request(
            path="/v1/search",
            payload={"query": query, "top_k": top_k},
        )


__all__ = ["WebSearchTool"]
