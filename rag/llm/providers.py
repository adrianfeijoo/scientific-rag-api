"""Minimal HTTP chat clients for the supported LLM providers.

Providers are called through their public REST APIs with httpx instead of
vendor SDKs: one shared dependency, a uniform `complete()` interface, and
the same reachability logic reused for auto-detection (Ollama). Freezing
the environment also stays much simpler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from rag.exceptions import LlmError


@dataclass(frozen=True)
class LlmInfo:
    provider: str  # "openai" | "anthropic" | "deepseek" | "ollama"
    model: str


class BaseChatClient(ABC):
    """Single-shot chat completion contract shared by every provider."""

    def __init__(self, info: LlmInfo, *, timeout: float, max_tokens: int) -> None:
        self.info = info
        self._timeout = timeout
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, *, temperature: float) -> str:
        try:
            return self._request(system, user, temperature=temperature)
        except LlmError:
            raise
        except Exception as exc:  # httpx transport errors, malformed payloads
            raise LlmError(f"{self.info.provider} request failed: {exc}") from exc

    @abstractmethod
    def _request(self, system: str, user: str, *, temperature: float) -> str: ...

    def _post_json(self, url: str, *, headers: dict[str, str], payload: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            # Truncated body: provider error payloads can be huge HTML pages.
            raise LlmError(
                f"{self.info.provider} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()


class OpenAiCompatibleClient(BaseChatClient):
    """OpenAI chat completions API; DeepSeek is API-compatible with a different base URL."""

    def __init__(
        self,
        info: LlmInfo,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float,
        max_tokens: int,
    ) -> None:
        super().__init__(info, timeout=timeout, max_tokens=max_tokens)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _request(self, system: str, user: str, *, temperature: float) -> str:
        data = self._post_json(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={
                "model": self.info.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": self._max_tokens,
            },
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected {self.info.provider} response shape: {data}") from exc


class AnthropicClient(BaseChatClient):
    """Anthropic Messages API."""

    _API_VERSION = "2023-06-01"

    def __init__(
        self, info: LlmInfo, *, api_key: str, timeout: float, max_tokens: int
    ) -> None:
        super().__init__(info, timeout=timeout, max_tokens=max_tokens)
        self._api_key = api_key

    def _request(self, system: str, user: str, *, temperature: float) -> str:
        data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._API_VERSION,
            },
            payload={
                "model": self.info.model,
                # max_tokens is mandatory in the Messages API.
                "max_tokens": self._max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
            },
        )
        try:
            return "".join(
                block["text"] for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LlmError(f"Unexpected anthropic response shape: {data}") from exc


class OllamaClient(BaseChatClient):
    """Local Ollama daemon through its REST API (no API key needed)."""

    def __init__(
        self,
        info: LlmInfo,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float,
        max_tokens: int,
    ) -> None:
        super().__init__(info, timeout=timeout, max_tokens=max_tokens)
        self._base_url = base_url.rstrip("/")

    def _request(self, system: str, user: str, *, temperature: float) -> str:
        data = self._post_json(
            f"{self._base_url}/api/chat",
            headers={},
            payload={
                "model": self.info.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": self._max_tokens,
                },
            },
        )
        try:
            return data["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise LlmError(f"Unexpected ollama response shape: {data}") from exc
