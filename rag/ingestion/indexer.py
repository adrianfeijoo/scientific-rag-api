"""Embedding of chunks for dense retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag.ingestion.chunker import Chunk

if TYPE_CHECKING:
    from rag.retrieval.embeddings import Embedder


def embedding_text(chunk: Chunk, *, with_section: bool = True) -> str:
    """Text fed to the dense encoder.

    Chroma metadata is invisible to the embedding model, so prepending the
    section title (a cheap, local version of contextual retrieval) lets a
    query like "sample preparation" match a chunk whose body never repeats
    those words. The stored `content` stays untouched for display and
    citations; only the vector sees the prefixed text.
    """
    if with_section and chunk.section:
        return f"{chunk.section}\n{chunk.content}"
    return chunk.content


def embed_chunks(
    chunks: list[Chunk],
    embedder: "Embedder",
    *,
    with_section: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode all chunks in batches (sentence-transformers shows its own bar)."""
    texts = [embedding_text(chunk, with_section=with_section) for chunk in chunks]
    return embedder.embed_passages(texts, batch_size=batch_size, show_progress_bar=True)
