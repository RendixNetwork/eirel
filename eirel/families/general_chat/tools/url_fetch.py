from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class UrlFetchTool(GeneralChatTool):
    """Open a public URL and return readable text + metadata.

    Mirrors ChatGPT's ``browser`` and Claude's ``web_fetch`` tool: input
    is a URL, output is extracted page content the LLM can quote or
    summarize. The owner-api-hosted service handles SSRF protection,
    per-host rate limiting, response-size caps, and HTML→text extraction
    so the miner agent never reaches the public web directly.
    """

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "url_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a single public URL and return its main text content, "
            "title, outbound links, and HTTP metadata. Use after web_search "
            "when you need to read a specific page in detail. The service "
            "blocks private/internal addresses, caps response size, and "
            "rate-limits per host."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s):// URL of the page to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 500_000,
                    "default": 50_000,
                    "description": (
                        "Hard cap on returned content length in characters. "
                        "Body bytes beyond the configured server-side cap "
                        "(default 1 MB) are dropped before extraction."
                    ),
                },
                "include_links": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Whether to include up to 50 outbound links from the "
                        "page in the response."
                    ),
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        url = str(kwargs["url"])
        max_chars = int(kwargs.get("max_chars", 50_000))
        include_links = bool(kwargs.get("include_links", True))
        return await self._client.request(
            path="/v1/fetch",
            payload={
                "url": url,
                "max_chars": max_chars,
                "include_links": include_links,
            },
        )


__all__ = ["UrlFetchTool"]
