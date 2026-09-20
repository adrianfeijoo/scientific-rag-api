"""POST /query: full RAG cycle (retrieval + LLM synthesis + citations)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.dependencies import get_llm, get_retriever
from app.schemas import QueryRequest, QueryResponse
from rag.llm.providers import BaseChatClient, LlmInfo
from rag.pipeline import RagPipeline
from rag.retrieval.hybrid import HybridRetriever

router = APIRouter(tags=["RAG"])

NO_LLM_MESSAGE = "No LLM provider available. Set an API key or use the /retrieve endpoint."


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question with an LLM over retrieved, cited sources",
)
def run_query(
    payload: QueryRequest,
    request: Request,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    llm: Annotated[
        tuple[BaseChatClient | None, LlmInfo | None], Depends(get_llm)
    ],
) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question field cannot be empty.")
    client, info = llm
    if client is None:
        raise HTTPException(status_code=503, detail=NO_LLM_MESSAGE)

    source = payload.filter_source.strip() if payload.filter_source else None
    pipeline = RagPipeline(retriever)
    result = pipeline.answer(
        question=question,
        client=client,
        info=info,
        top_k=payload.top_k,
        source=source or None,
        temperature=payload.temperature,
    )
    return QueryResponse(question=question, **result)
