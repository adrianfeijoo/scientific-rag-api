"""Central configuration.

Every runtime knob (paths, chunking, retrieval, LLM providers) is read from
environment variables, optionally provided through a `.env` file at the
project root. Real environment variables always take precedence over
`.env` values, which is pydantic-settings' default and keeps 12-factor
deployments working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `app` package; anchors all default paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("*", mode="before")
    @classmethod
    def _empty_string_means_unset(cls, value: object, info: ValidationInfo) -> object:
        """Treat empty values in `.env` (e.g. `OLLAMA_BASE_URL=`) as unset.

        Without this, an empty string silently overrides the field default —
        for instance breaking Ollama auto-detection even with the server
        running. Copying `.env.example` and leaving keys blank must be safe.
        """
        if isinstance(value, str) and not value.strip():
            field = cls.model_fields.get(info.field_name)
            if field is not None and not field.is_required():
                return field.get_default(call_default_factory=True)
        return value

    # Paths
    pdf_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "scientific_docs"

    # Chunking (phase 2: recursive character split)
    # ~750 chars fits one dense paragraph or table block and stays well inside
    # bge-small's 512-token window; the overlap keeps formulas intact.
    chunk_size: int = 750
    chunk_overlap: int = 140

    # Embeddings / retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embed_section_context: bool = True  # prepend section title to embedded passage
    bge_query_instruction: bool = True  # BGE query-side prefix for short queries
    rrf_k: int = 60  # standard Reciprocal Rank Fusion constant
    candidate_multiplier: int = 3  # candidate pool per retriever = top_k * this

    # LLM
    # auto: first usable of openai -> anthropic -> deepseek -> ollama.
    llm_provider: Literal["auto", "openai", "anthropic", "deepseek", "ollama"] = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_timeout_seconds: float = 60.0
    # Local inference is CPU-bound and needs a much larger budget than APIs.
    ollama_timeout_seconds: float = 300.0
    llm_max_tokens: int = 1024
