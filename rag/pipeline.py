"""End-to-end RAG orchestration: retrieve -> prompt -> generate -> cite.
"""

from __future__ import annotations

import time

from rag.llm.prompts import SYSTEM_PROMPT, build_user_prompt, parse_citations
from rag.llm.providers import BaseChatClient, LlmInfo
from rag.retrieval.hybrid import HybridRetriever, RetrievedChunk


class RagPipeline:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def answer(
        self,
        *,
        question: str,
        client: BaseChatClient,
        info: LlmInfo,
        top_k: int,
        source: str | None,
        temperature: float,
    ) -> dict:
        """Run the full cycle and return the QueryResponse payload fields."""
        started = time.perf_counter()

        chunks = self._retriever.search(question, top_k=top_k, source=source)
        prompt = build_user_prompt(question, chunks)
        answer_text = client.complete(SYSTEM_PROMPT, prompt, temperature=temperature)

        sources = [
            {
                "citation_id": citation_id,
                "chunk_id": chunk.chunk_id,
                "document": chunk.source,
                "title": chunk.title,
                "pages": chunk.pages,
                "section": chunk.section,
            }
            # Ignore hallucinated markers pointing outside the provided sources.
            for citation_id in parse_citations(answer_text)
            if (chunk := _chunk_by_citation(chunks, citation_id)) is not None
        ]
        return {
            "answer": answer_text,
            "sources": sources,
            "llm_used": info.model,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def _chunk_by_citation(chunks: list[RetrievedChunk], citation_id: int) -> RetrievedChunk | None:
    if 1 <= citation_id <= len(chunks):
        return chunks[citation_id - 1]
    return None
