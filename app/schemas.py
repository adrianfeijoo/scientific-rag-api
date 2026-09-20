"""API request/response models: the source of truth for the OpenAPI schema."""

from __future__ import annotations

from pydantic import BaseModel, Field

# /retrieve


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Question or search terms.")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve.")
    filter_source: str | None = Field(
        None,
        description="Restrict the search to one PDF filename, e.g. 'sensors-24-01916.pdf'.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(
        ..., description="Stable id encoding source file, pages and chunk position."
    )
    content: str
    source: str
    title: str = Field(..., description="Article title extracted from the PDF.")
    pages: list[int]
    section: str = Field(
        ...,
        description="Section path from the markdown headers, e.g. 'Materials and Methods > Study Site'.",
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity between the query and the chunk embedding."
    )


class RetrieveResponse(BaseModel):
    query: str
    chunks_retrieved: int
    results: list[RetrievedChunk]


# /query


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question.")
    top_k: int = Field(4, ge=1, le=10, description="Number of chunks used as context.")
    filter_source: str | None = Field(
        None, description="Restrict the context to one PDF filename."
    )
    temperature: float = Field(
        0.0, ge=0.0, le=1.0, description="LLM sampling temperature; 0.0 for deterministic answers."
    )


class CitationSource(BaseModel):
    citation_id: int = Field(
        ..., description="Marker used in the answer text, e.g. 1 for '[1]'."
    )
    chunk_id: str
    document: str
    title: str
    pages: list[int]
    section: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[CitationSource]
    llm_used: str = Field(
        ..., description="Model identifier of the LLM that produced the answer."
    )
    latency_seconds: float


# /health


class HealthResponse(BaseModel):
    status: str
    vector_store: str = Field(
        ..., description="'connected', or 'empty' when ingestion has not run yet."
    )
    documents_indexed: int
    total_chunks: int
    active_llm_provider: str = Field(
        ..., description="Provider name, or 'none' when no LLM is configured."
    )
