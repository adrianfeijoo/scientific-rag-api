"""FastAPI application factory and lifespan wiring.

Heavy resources (embedding model, Chroma client, BM25 index, LLM client)
are built once at startup and shared through `app.state`; route handlers
read them per request via the dependencies in `app.dependencies`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.routers import health, query, retrieve
from rag.exceptions import (
    EmptyStoreError,
    LlmError,
    UnknownSourceError,
    VectorStoreError,
)
from rag.llm.factory import build_llm
from rag.retrieval.embeddings import get_embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.store import VectorStore

logger = logging.getLogger("scientific_rag")

API_TITLE = "Scientific RAG API"
API_DESCRIPTION = (
    "Minimal retrieval-augmented generation pipeline over a corpus of "
    "scientific PDFs: two-phase markdown-aware chunking, hybrid BM25 + dense "
    "retrieval fused with RRF, and multi-provider LLM answers with citations."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Loading embedding model '%s'...", settings.embedding_model)
    embedder = get_embedder(
        settings.embedding_model, query_instruction=settings.bge_query_instruction
    )

    store = VectorStore(settings.chroma_dir, settings.collection_name)
    retriever = HybridRetriever(
        store,
        embedder,
        rrf_k=settings.rrf_k,
        candidate_multiplier=settings.candidate_multiplier,
    )

    # (None, None) in auto mode when nothing is configured: /query then
    # answers 503 and re-probes lazily (see app.dependencies.get_llm).
    llm_client, llm_info = build_llm(settings)

    app.state.settings = settings
    app.state.retriever = retriever
    app.state.llm_client = llm_client
    app.state.llm_info = llm_info
    # 0.0 = "never probed": the first /query after boot may retry immediately.
    app.state.llm_last_probe = 0.0

    logger.info(
        "Startup complete: %d chunks from %d documents | LLM provider: %s",
        retriever.total_chunks,
        len(retriever.sources),
        llm_info.provider if llm_info else "none",
    )
    yield
    # The persistent Chroma client flushes on every write; nothing to close.


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(retrieve.router)
    app.include_router(query.router)

    @app.get("/", tags=["Meta"], summary="Service info")
    def root() -> dict:
        return {
            "service": API_TITLE,
            "endpoints": {
                "health": "GET /health",
                "retrieve": "POST /retrieve",
                "query": "POST /query",
                "docs": "GET /docs",
            },
        }

    # Domain exceptions -> HTTP semantics, keeping route handlers clean.

    @app.exception_handler(EmptyStoreError)
    async def empty_store_handler(request: Request, exc: EmptyStoreError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(UnknownSourceError)
    async def unknown_source_handler(
        request: Request, exc: UnknownSourceError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(VectorStoreError)
    async def vector_store_handler(
        request: Request, exc: VectorStoreError
    ) -> JSONResponse:
        logger.error("Vector store failure: %s", exc)
        return JSONResponse(
            status_code=500, content={"detail": "Internal error processing vector store."}
        )

    @app.exception_handler(LlmError)
    async def llm_handler(request: Request, exc: LlmError) -> JSONResponse:
        logger.error("LLM provider failure: %s", exc)
        return JSONResponse(
            status_code=502, content={"detail": f"LLM provider error: {exc}"}
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
