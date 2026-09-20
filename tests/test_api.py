"""API integration tests.

These exercise the real stack (embedding model + Chroma + hybrid retrieval)
and therefore require a completed ingestion run. The /query happy path uses
a fake LLM client so it stays deterministic and needs no keys or network.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from rag.llm.providers import BaseChatClient, LlmInfo

SETTINGS = Settings()
_STORE_READY = SETTINGS.chroma_dir.exists() and any(SETTINGS.chroma_dir.iterdir())

pytestmark = pytest.mark.skipif(
    not _STORE_READY, reason="vector store is empty: run scripts/ingest.py first"
)

QUESTION = "Which vegetation index is best for wheat water status under mildew?"
NO_LLM_MESSAGE = "No LLM provider available. Set an API key or use the /retrieve endpoint."


class FakeLlmClient(BaseChatClient):
    """Fixed answer so /query is tested offline and deterministically."""

    def __init__(self) -> None:
        super().__init__(
            LlmInfo(provider="fake", model="fake-model-for-tests"),
            timeout=1.0,
            max_tokens=64,
        )

    def _request(self, system: str, user: str, *, temperature: float) -> str:
        # Sanity-check the prompt contract the citation parser relies on.
        assert "Source [1]" in user and "Question:" in user
        return (
            "The Photochemical Reflectance Index (PRI) is the most sensitive index [1], "
            "while water indices such as WBI perform poorly [2]. "
            "A hallucinated marker [9] must be ignored."
        )


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    body = client.get("/").json()
    assert body["endpoints"]["query"] == "POST /query"


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["vector_store"] == "connected"
    assert body["documents_indexed"] == 5
    assert body["total_chunks"] > 50
    assert body["active_llm_provider"] in {"openai", "anthropic", "deepseek", "ollama", "none"}


def test_retrieve_returns_ranked_chunks(client):
    body = client.post("/retrieve", json={"query": QUESTION, "top_k": 4}).json()
    assert body["query"] == QUESTION
    assert body["chunks_retrieved"] == len(body["results"]) == 4
    top = body["results"][0]
    for field in ("chunk_id", "content", "source", "title", "pages", "section", "similarity_score"):
        assert field in top
    assert -1.0 <= top["similarity_score"] <= 1.0
    # Top result must be on-topic for this corpus.
    assert top["source"] == "fpls-08-01219.pdf"


def test_retrieve_supports_source_filter(client):
    body = client.post(
        "/retrieve", json={"query": "vegetation index", "filter_source": "fpls-08-01219.pdf"}
    ).json()
    assert body["results"]
    assert all(result["source"] == "fpls-08-01219.pdf" for result in body["results"])


def test_retrieve_unknown_source_returns_404(client):
    response = client.post("/retrieve", json={"query": "wheat", "filter_source": "nope.pdf"})
    assert response.status_code == 404
    assert "Available sources" in response.json()["detail"]


def test_retrieve_blank_query_returns_400(client):
    response = client.post("/retrieve", json={"query": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "The query field cannot be empty."


def test_retrieve_invalid_top_k_returns_422(client):
    assert client.post("/retrieve", json={"query": "wheat", "top_k": 11}).status_code == 422
    assert client.post("/retrieve", json={"query": "wheat", "top_k": "many"}).status_code == 422


def test_query_end_to_end_with_fake_llm(client, app):
    original = (app.state.llm_client, app.state.llm_info)
    app.state.llm_client = FakeLlmClient()
    app.state.llm_info = app.state.llm_client.info
    try:
        response = client.post("/query", json={"question": QUESTION, "top_k": 4})
        assert response.status_code == 200
        body = response.json()
        assert body["question"] == QUESTION
        assert "PRI" in body["answer"]
        assert body["llm_used"] == "fake-model-for-tests"
        assert body["latency_seconds"] >= 0
        # [9] points outside the provided sources and must be dropped.
        assert [source["citation_id"] for source in body["sources"]] == [1, 2]
        for source in body["sources"]:
            for field in ("chunk_id", "document", "title", "pages", "section"):
                assert source[field]
    finally:
        app.state.llm_client, app.state.llm_info = original


def test_query_without_llm_returns_503(client, app):
    original = (app.state.llm_client, app.state.llm_info, app.state.llm_last_probe)
    app.state.llm_client = None
    app.state.llm_info = None
    # Fresh probe timestamp suppresses lazy re-detection for 30 s.
    app.state.llm_last_probe = time.monotonic()
    try:
        response = client.post("/query", json={"question": QUESTION})
        assert response.status_code == 503
        assert response.json()["detail"] == NO_LLM_MESSAGE
    finally:
        app.state.llm_client, app.state.llm_info, app.state.llm_last_probe = original


def test_query_blank_question_returns_400(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400


def test_query_invalid_temperature_returns_422(client):
    response = client.post("/query", json={"question": QUESTION, "temperature": 1.5})
    assert response.status_code == 422
