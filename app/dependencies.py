"""Dependency wiring between the HTTP layer and the rag package."""

from __future__ import annotations

import time

from fastapi import Request

from app.config import Settings
from rag.llm.factory import build_llm
from rag.llm.providers import BaseChatClient, LlmInfo
from rag.retrieval.hybrid import HybridRetriever

# When no provider was found, wait this long before probing again: the probe
# is an Ollama ping with a 2 s timeout, and doing that on every /query call
# while unconfigured would be needlessly slow.
LLM_PROBE_INTERVAL_SECONDS = 30.0


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_llm(request: Request) -> tuple[BaseChatClient | None, LlmInfo | None]:
    """Return the active LLM client, lazily re-detecting when none was found.

    This lets an operator start Ollama (or fix the environment) after the API
    booted and have /query work without a restart.
    """
    state = request.app.state
    if state.llm_client is None:
        now = time.monotonic()
        if now - state.llm_last_probe >= LLM_PROBE_INTERVAL_SECONDS:
            state.llm_last_probe = now
            client, info = build_llm(state.settings)
            if client is not None:
                state.llm_client, state.llm_info = client, info
    return state.llm_client, state.llm_info
