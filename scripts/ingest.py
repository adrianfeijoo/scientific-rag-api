#!/usr/bin/env python3
"""Ingest the corpus PDFs into the local vector store.

Run from the project root:

    python scripts/ingest.py

The collection is fully rebuilt on every run, which keeps ingestion
idempotent and deterministic (chunking parameters can change between runs
without leaving stale chunks behind).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Plain-script execution: make the project-root packages (app, rag) importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from app.config import Settings
from rag.ingestion.chunker import chunk_document
from rag.ingestion.indexer import embed_chunks
from rag.ingestion.loader import load_pdf
from rag.retrieval.embeddings import get_embedder
from rag.retrieval.store import VectorStore


def main() -> int:
    settings = Settings()
    pdf_paths = sorted(settings.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {settings.pdf_dir}.", file=sys.stderr)
        return 1

    print(f"Found {len(pdf_paths)} PDFs in {settings.pdf_dir}")
    print(f"Embedding model: {settings.embedding_model} (downloaded from Hugging Face on first run)")

    # Heavy resources are created once and reused across all files.
    embedder = get_embedder(
        settings.embedding_model, query_instruction=settings.bge_query_instruction
    )
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    store.reset()

    chunks = []
    for pdf_path in tqdm(pdf_paths, desc="Parsing & chunking", unit="pdf"):
        loaded = load_pdf(pdf_path)
        chunks.extend(
            chunk_document(
                loaded.pages,
                source=loaded.source,
                title=loaded.title,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    if not chunks:
        print("No chunks were produced from the corpus.", file=sys.stderr)
        return 1

    per_file: dict[str, int] = {}
    for chunk in chunks:
        per_file[chunk.source] = per_file.get(chunk.source, 0) + 1
    for source, count in sorted(per_file.items()):
        print(f"  {source}: {count} chunks")
    print(
        f"Total: {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap})"
    )

    # sentence-transformers renders its own progress bar for the slow step.
    embeddings = embed_chunks(chunks, embedder, with_section=settings.embed_section_context)
    store.upsert_chunks(chunks, embeddings, show_progress=True)

    print(
        f"Indexed {store.count()} chunks into {settings.chroma_dir} "
        f"(collection '{settings.collection_name}')"
    )
    print("Start the API with: uvicorn app.main:app --reload   (see README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
