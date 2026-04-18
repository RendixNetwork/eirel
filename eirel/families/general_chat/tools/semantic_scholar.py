from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class SemanticScholarTool(GeneralChatTool):
    """Owner-api-routed Semantic Scholar search.

    Covers peer-reviewed papers and arXiv preprints in one index — the
    primary scientific search tool for general_chat agents. Miners can
    still use the general web search tool for science queries if they
    prefer that strategy; this tool is not mandatory for science tasks.
    """

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "semantic_scholar"

    @property
    def description(self) -> str:
        return (
            "Search peer-reviewed papers and arXiv preprints via the "
            "owner-hosted Semantic Scholar tool service. Returns title, "
            "abstract, authors, year, venue, citation count, arXiv id, "
            "DOI, and open-access PDF URL per result. Useful for "
            "academic and scientific queries."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or natural language).",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                    "description": "Maximum number of results to return.",
                },
                "year": {
                    "type": "string",
                    "description": (
                        "Optional year filter, e.g. '2023-' for 2023 "
                        "onwards or '2020-2024' for a range."
                    ),
                },
                "fields_of_study": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter by field of study, e.g. "
                        "['Computer Science', 'Biology']."
                    ),
                },
                "open_access_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, only return papers with an open-access PDF.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs["query"])
        max_results = int(kwargs.get("max_results", 5))
        max_results = max(1, min(max_results, 100))
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        year = kwargs.get("year")
        if isinstance(year, str) and year.strip():
            payload["year"] = year.strip()
        fields_of_study = kwargs.get("fields_of_study")
        if isinstance(fields_of_study, list) and fields_of_study:
            payload["fields_of_study"] = [str(f) for f in fields_of_study]
        if kwargs.get("open_access_only") is True:
            payload["open_access_only"] = True
        return await self._client.request(
            path="/v1/tools/semantic_scholar",
            payload=payload,
        )


__all__ = ["SemanticScholarTool"]
