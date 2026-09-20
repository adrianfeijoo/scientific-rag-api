"""Singleton access to the sentence-transformers encoder.

A single model instance is shared by ingestion and the API: loading the
weights takes seconds and must not happen per request.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE encoders are trained with a query-side instruction prefix; it improves
# short-query -> passage retrieval. Passages are embedded without it.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embedders: dict[str, Embedder] = {}


class Embedder:
    """Unit-normalised encoder: with normalisation, dot product == cosine."""

    def __init__(self, model_name: str, *, query_instruction: bool = True) -> None:
        self._model = SentenceTransformer(model_name)
        uses_bge = "bge" in model_name.lower()
        self._query_prefix = _BGE_QUERY_INSTRUCTION if query_instruction and uses_bge else ""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query; returns a 1D unit vector."""
        return self._model.encode(
            [self._query_prefix + text], normalize_embeddings=True
        )[0]

    def embed_passages(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed passages; returns a 2D array of unit vectors."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )


def get_embedder(model_name: str, *, query_instruction: bool = True) -> Embedder:
    key = f"{model_name}|{query_instruction}"
    if key not in _embedders:
        _embedders[key] = Embedder(model_name, query_instruction=query_instruction)
    return _embedders[key]
