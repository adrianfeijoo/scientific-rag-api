"""GET /health: liveness/readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas import HealthResponse
from rag.exceptions import VectorStoreError

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service status")
def health(request: Request) -> HealthResponse:
    retriever = request.app.state.retriever
    llm_info = request.app.state.llm_info
    try:
        total_chunks = retriever.total_chunks
        documents_indexed = len(retriever.sources)
    except VectorStoreError:  # pragma: no cover - defensive: Chroma dir unreadable/corrupt
        raise HTTPException(
            status_code=503, detail="Vector store is not responding."
        ) from None
    return HealthResponse(
        status="healthy",
        vector_store="connected" if total_chunks > 0 else "empty",
        documents_indexed=documents_indexed,
        total_chunks=total_chunks,
        active_llm_provider=llm_info.provider if llm_info else "none",
    )
