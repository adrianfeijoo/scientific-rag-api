"""LLM provider selection and construction from settings.

``auto`` (the default) picks the first usable provider in the order
openai -> anthropic -> deepseek -> ollama, so a machine with no API keys
and no local server degrades to "no LLM" (503 on /query) instead of
crashing. An explicitly configured provider fails fast at startup with a
clear message: silent fallbacks would hide operator mistakes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from rag.exceptions import ConfigurationError
from rag.llm.providers import (
    AnthropicClient,
    BaseChatClient,
    LlmInfo,
    OllamaClient,
    OpenAiCompatibleClient,
)

if TYPE_CHECKING:
    # Typing-only import: the runtime dependency direction is app -> rag.
    from app.config import Settings

logger = logging.getLogger(__name__)

_OLLAMA_PROBE_TIMEOUT_SECONDS = 2.0
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_llm(settings: "Settings") -> tuple[BaseChatClient | None, LlmInfo | None]:
    """Return (client, info) for the configured provider, or (None, None) in
    auto mode when nothing is usable."""
    if settings.llm_provider == "auto":
        for provider in ("openai", "anthropic", "deepseek", "ollama"):
            if provider == "ollama":
                if not _ollama_reachable(settings.ollama_base_url):
                    continue
            elif not _api_key(settings, provider):
                continue
            logger.info("Auto-detected LLM provider: %s", provider)
            return _build_provider(settings, provider)
        logger.info("No LLM provider available; /query will return 503 until one is configured")
        return None, None
    return _build_provider(settings, settings.llm_provider)


def _api_key(settings: "Settings", provider: str) -> str | None:
    return {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "deepseek": settings.deepseek_api_key,
    }.get(provider)


def _build_provider(settings: "Settings", provider: str) -> tuple[BaseChatClient, LlmInfo]:
    if provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
        info = LlmInfo("openai", settings.openai_model)
        return (
            OpenAiCompatibleClient(
                info,
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
            ),
            info,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY.")
        info = LlmInfo("anthropic", settings.anthropic_model)
        return (
            AnthropicClient(
                info,
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
            ),
            info,
        )
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ConfigurationError("LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY.")
        info = LlmInfo("deepseek", settings.deepseek_model)
        return (
            OpenAiCompatibleClient(
                info,
                api_key=settings.deepseek_api_key,
                base_url=_DEEPSEEK_BASE_URL,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
            ),
            info,
        )
    if provider == "ollama":
        if not _ollama_reachable(settings.ollama_base_url):
            raise ConfigurationError(
                f"LLM_PROVIDER=ollama requires a running Ollama server at "
                f"{settings.ollama_base_url} (start it with: ollama serve)."
            )
        info = LlmInfo("ollama", settings.ollama_model)
        return (
            OllamaClient(
                info,
                base_url=settings.ollama_base_url,
                # Local inference is often CPU-bound and much slower than APIs.
                timeout=settings.ollama_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
            ),
            info,
        )
    raise ConfigurationError(
        f"Unknown LLM_PROVIDER '{provider}'. Use auto, openai, anthropic, deepseek or ollama."
    )


def _ollama_reachable(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=_OLLAMA_PROBE_TIMEOUT_SECONDS) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
        return response.status_code == 200
    except Exception:  # noqa: BLE001 - unreachable means "not available", not an error
        return False
