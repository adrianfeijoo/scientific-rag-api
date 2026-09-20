"""Persistent ChromaDB wrapper (local-first: on-disk, embedded, no server).

Cosine space is used so chunk distance maps directly to cosine similarity;
embeddings are unit-normalised on both the indexing and the query side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

from rag.exceptions import VectorStoreError
from rag.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class VectorStore:
    def __init__(self, chroma_dir: Path, collection_name: str) -> None:
        self._name = collection_name
        try:
            # Telemetry off: the system must work fully offline and not phone home.
            self._client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name, metadata=_COLLECTION_METADATA
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open vector store at {chroma_dir}: {exc}") from exc

    def reset(self) -> None:
        """Drop and recreate the collection: ingestion is a full, idempotent rebuild."""
        try:
            self._client.delete_collection(self._name)
            logger.info("Dropped existing collection '%s'", self._name)
        except Exception:
            # A missing collection is the normal first-run case, not an error.
            logger.debug("Collection '%s' did not exist yet; creating it", self._name, exc_info=True)
        self._collection = self._client.create_collection(
            name=self._name, metadata=_COLLECTION_METADATA
        )

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Vector store count failed: {exc}") from exc

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings,
        *,
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> None:
        try:
            starts = range(0, len(chunks), batch_size)
            if show_progress:
                starts = tqdm(list(starts), desc="Indexing chunks", unit="batch")
            for start in starts:
                window = chunks[start : start + batch_size]
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in window],
                    documents=[chunk.content for chunk in window],
                    metadatas=[_chunk_metadata(chunk) for chunk in window],
                    embeddings=embeddings[start : start + batch_size].tolist(),
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks: {exc}") from exc

    def query(self, query_embedding, *, k: int, where: dict | None = None) -> list[dict]:
        """Nearest chunks by cosine distance, ordered best first."""
        try:
            response = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store query failed: {exc}") from exc
        return [
            {
                "chunk_id": chunk_id,
                "content": document,
                "metadata": metadata or {},
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                response["ids"][0],
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
            )
        ]

    def get_all(self) -> list[dict]:
        """Full dump; used to build the in-memory BM25 index (small corpus)."""
        try:
            response = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Vector store read failed: {exc}") from exc
        return [
            {"chunk_id": chunk_id, "content": document, "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(
                response["ids"], response["documents"], response["metadatas"]
            )
        ]

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        try:
            response = self._collection.get(
                ids=chunk_ids, include=["documents", "metadatas", "embeddings"]
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store read failed: {exc}") from exc
        return {
            chunk_id: {"content": document, "metadata": metadata or {}, "embedding": embedding}
            for chunk_id, document, metadata, embedding in zip(
                response["ids"],
                response["documents"],
                response["metadatas"],
                response["embeddings"],
            )
        }


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    # Chroma metadata values must be scalars, so the page range is stored as
    # a start/end pair and rebuilt as a list at the API layer.
    return {
        "source": chunk.source,
        "title": chunk.title,
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
    }
