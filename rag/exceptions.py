"""Domain exceptions.

The `rag` package raises these; the FastAPI layer maps each one to a
specific HTTP status so route handlers stay free of try/except noise.
"""


class EmptyStoreError(RuntimeError):
    """A query arrived before any document was indexed (ingestion pending)."""


class UnknownSourceError(ValueError):
    """A metadata filter references a file that was never indexed."""

    def __init__(self, source: str, available: list[str]) -> None:
        self.source = source
        self.available = available
        super().__init__(
            f"Unknown source file '{source}'. Available sources: {', '.join(sorted(available))}"
        )


class VectorStoreError(RuntimeError):
    """The underlying vector store failed (I/O error, corruption, ...)."""


class LlmError(RuntimeError):
    """The configured LLM provider failed or was unreachable."""


class ConfigurationError(ValueError):
    """An explicitly configured provider is unusable (fail fast at startup)."""
