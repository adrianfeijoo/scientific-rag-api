"""POST /retrieve: hybrid search over the indexed chunks (no LLM involved)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_retriever
from app.schemas import RetrievedChunk as RetrievedChunkSchema
from app.schemas import RetrieveRequest, RetrieveResponse
from rag.retrieval.hybrid import HybridRetriever

router = APIRouter(tags=["Retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Hybrid (dense + BM25) search, auditable without any LLM",
)
def retrieve(
    payload: RetrieveRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> RetrieveResponse:
    query = payload.query.strip()
    if not query:
        # 400 rather than 422: the value is well-typed, just semantically empty.
        raise HTTPException(status_code=400, detail="The query field cannot be empty.")
    source = payload.filter_source.strip() if payload.filter_source else None

    chunks = retriever.search(query, top_k=payload.top_k, source=source or None)
    return RetrieveResponse(
        query=query,
        chunks_retrieved=len(chunks),
        results=[
            RetrievedChunkSchema(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                title=chunk.title,
                pages=chunk.pages,
                section=chunk.section,
                similarity_score=round(chunk.similarity_score, 4),
            )
            for chunk in chunks
        ],
    )
