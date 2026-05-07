from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class RagTool(GeneralChatTool):
    """Retrieve top-k chunks from a corpus indexed on the owner side.

    Mirrors the Claude Projects / ChatGPT Projects retrieval pattern:
    the user (product mode) or the operator (eval mode) has indexed
    one or more documents into a corpus identified by ``corpus_id``;
    the agent issues a natural-language query and gets back ranked
    chunks of source text it can quote or summarize.

    The service is server-attested — every retrieve call is logged in
    the orchestrator's tool-call ledger, so the validator can score
    "did the miner actually retrieve before answering?" from authoritative
    records (not from miner-emitted trace frames).

    Usage in a graph node typically looks like:

        result = await rag.execute(
            corpus_id="project-XYZ",
            query="What is the refund policy?",
            k=5,
        )
        chunks = result["chunks"]  # list of {doc_id, chunk_id, text, score}
    """

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "rag.retrieve"

    @property
    def description(self) -> str:
        return (
            "Retrieve top-k passages from a curated document corpus by "
            "natural-language query. Use this for grounded answers when "
            "the task references an attached document set, project, or "
            "knowledge base — the corpus_id is provided in the task's "
            "metadata. Returns ranked chunks with doc_id and similarity "
            "score so you can quote sources accurately. Prefer this over "
            "trying to fit long documents in your context window."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "corpus_id": {
                    "type": "string",
                    "description": (
                        "Identifier of the corpus to search. Provided "
                        "to the agent via the task's "
                        "metadata.rag_corpus_id field."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query. Specific, content-rich "
                        "queries retrieve better than broad ones."
                    ),
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": (
                        "Number of chunks to retrieve. 3-5 is usually "
                        "enough; higher k dilutes precision."
                    ),
                },
            },
            "required": ["corpus_id", "query"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        corpus_id = str(kwargs["corpus_id"])
        query = str(kwargs["query"])
        k = int(kwargs.get("k", 5))
        return await self._client.request(
            path="/v1/rag/retrieve",
            payload={"corpus_id": corpus_id, "query": query, "k": k},
        )
