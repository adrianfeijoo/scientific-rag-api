"""Prompt construction and citation parsing.

The [n] markers in the prompt are the contract between the LLM and the
citation parser: the number given to a source here is exactly the
`citation_id` the API returns, so every citation resolves to one chunk.
"""

from __future__ import annotations

import re

from rag.retrieval.hybrid import RetrievedChunk

SYSTEM_PROMPT = """You are a rigorous scientific literature assistant.

Answer the user's question using ONLY the numbered sources provided in the context.

Rules:
- Cite the sources you rely on with bracketed markers like [1] or [2][4]. Cite every non-trivial claim.
- Use precise technical language and report key figures (R2, RMSE, p-values, ...) exactly as stated in the sources.
- If sources agree or disagree, make that explicit and cite all of them.
- If the context does not contain enough information, say so plainly; never invent facts or citations.
- Keep the answer focused (a few sentences up to a short paragraph) and write it in the same language as the question."""

_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "?"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render the retrieved chunks as numbered sources followed by the question."""
    lines = ["Context: excerpts retrieved from the indexed scientific papers.", ""]
    for position, chunk in enumerate(chunks, start=1):
        lines.extend(
            [
                "----------------",
                f"Source [{position}]",
                f"Chunk_ID: {chunk.chunk_id}",
                (
                    f"Document: {chunk.source} | Pages: {format_pages(chunk.pages)} | "
                    f"Section: {chunk.section or 'unknown'}"
                ),
                "Content:",
                chunk.content,
            ]
        )
    lines.extend(
        [
            "----------------",
            "",
            f"Question: {question}",
            "",
            "Answer (cite sources with [n] markers):",
        ]
    )
    return "\n".join(lines)


def parse_citations(answer: str) -> list[int]:
    """Citation ids in order of first appearance; supports [2] and [1,3]."""
    seen: dict[int, None] = {}  # dict preserves insertion order while staying unique
    for match in _CITATION_PATTERN.findall(answer):
        for part in match.split(","):
            seen[int(part)] = None
    return list(seen)
