"""Hybrid retrieval: dense (Chroma + bge) fused with sparse (BM25) via RRF.

Dense search captures paraphrases and topical similarity; BM25 keeps exact
terminology characteristic of scientific text (index names like "FWBI1",
species names, chemicals) that embeddings tend to blur. Reciprocal Rank
Fusion combines both rankings without requiring comparable score scales,
and keeps the system light: no cross-encoder reranking stage.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.exceptions import EmptyStoreError, UnknownSourceError
from rag.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; single characters are dropped.

    One-character tokens in this corpus are mostly fragments of formulas and
    page furniture ("R2 = 0.639" -> r, 2, 0, 639) and add no useful signal.
    """
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


def rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank).

    `k=60` is the de-facto standard constant; it dampens the weight of top
    positions so neither retriever can dominate the fusion alone.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    title: str
    section: str
    page_start: int
    page_end: int
    similarity_score: float  # cosine(query, chunk embedding) - reported to users
    fused_score: float       # RRF score - internal ordering signal only

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))


def _source_filter(source: str | None) -> dict | None:
    return None if source is None else {"source": {"$eq": source}}


class HybridRetriever:
    """Dense + sparse retrieval over a single Chroma collection.

    The BM25 index lives in memory and is rebuilt whenever the collection
    size changes (e.g. the ingestion script ran against the same directory),
    keeping the sparse side consistent without an API restart.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_sources: list[str] = []
        self._sources: set[str] = set()

    @property
    def total_chunks(self) -> int:
        return self._store.count()

    @property
    def sources(self) -> set[str]:
        self._refresh_if_needed()
        return set(self._sources)

    def search(
        self, query: str, *, top_k: int, source: str | None = None
    ) -> list[RetrievedChunk]:
        self._refresh_if_needed()
        if not self._corpus_ids:
            raise EmptyStoreError(
                "The vector store is empty. Run 'python scripts/ingest.py' before querying."
            )
        if source is not None and source not in self._sources:
            raise UnknownSourceError(source, sorted(self._sources))

        # Wider candidate pool than top_k: RRF rewards chunks that both
        # retrievers rank decently, which single-retriever top_k would miss.
        candidate_k = min(max(top_k * self._candidate_multiplier, 10), len(self._corpus_ids))

        query_embedding = self._embedder.embed_query(query)
        dense_ids = [
            record["chunk_id"]
            for record in self._store.query(
                query_embedding, k=candidate_k, where=_source_filter(source)
            )
        ]
        sparse_ids = self._sparse_search(query, source, candidate_k)

        fused = rrf_fuse([dense_ids, sparse_ids], k=self._rrf_k)
        # Fused score first, chunk id as a deterministic tie-breaker.
        top_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
        if not top_ids:
            return []

        records = self._store.get_by_ids(top_ids)
        results: list[RetrievedChunk] = []
        for chunk_id in top_ids:
            record = records.get(chunk_id)
            if record is None:
                continue  # chunk vanished between fusion and fetch
            metadata = record["metadata"]
            # Both sides are unit-normalised, so dot product == cosine.
            similarity = float(
                np.dot(np.asarray(record["embedding"], dtype=np.float32), query_embedding)
            )
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=record["content"],
                    source=str(metadata.get("source", "")),
                    title=str(metadata.get("title", "")),
                    section=str(metadata.get("section", "")),
                    page_start=int(metadata.get("page_start", 0)),
                    page_end=int(metadata.get("page_end", 0)),
                    similarity_score=similarity,
                    fused_score=fused[chunk_id],
                )
            )
        return results

    def _refresh_if_needed(self) -> None:
        total = self._store.count()
        if self._bm25 is not None and total == len(self._corpus_ids):
            return
        records = self._store.get_all()
        self._corpus_ids = [record["chunk_id"] for record in records]
        self._corpus_sources = [str(record["metadata"].get("source", "")) for record in records]
        self._sources = {s for s in self._corpus_sources if s}
        # BM25Okapi raises on an empty corpus; keep None and let the caller
        # see the empty-store condition instead.
        if self._corpus_ids:
            self._bm25 = BM25Okapi([tokenize(record["content"]) for record in records])
        else:
            self._bm25 = None
        logger.info(
            "Sparse index (re)built: %d chunks from %d documents",
            len(self._corpus_ids),
            len(self._sources),
        )

    def _sparse_search(self, query: str, source: str | None, k: int) -> list[str]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Stable descending order: equal-score ties resolve deterministically.
        order = np.argsort(-scores, kind="stable")
        ranking = [
            self._corpus_ids[position]
            for position in order
            if scores[position] > 0
            and (source is None or self._corpus_sources[position] == source)
        ]
        return ranking[:k]
