"""LLM providers.

Ollama is the default and the only one required. An OpenAI-compatible provider
is available for people who already run vLLM, LM Studio or similar, but nothing
falls back to it automatically: if the local model is unreachable the request
fails loudly. Silently shipping someone's unpublished paper to a hosted API
because their local server was down would be a serious breach of the promise
this project makes.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the model cannot be reached or returns nothing usable."""


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str: ...

    @abstractmethod
    def available(self) -> tuple[bool, str]: ...


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama daemon over HTTP."""

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout_seconds: int = 180,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> tuple[bool, str]:
        """Check the daemon is up and the configured model is pulled.

        Used to give a precise error at the point of asking, instead of a
        connection traceback: "Ollama is not running" and "you have Ollama but
        not this model" need different fixes.
        """
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any failure means unavailable
            return False, (
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`. ({exc})"
            )

        installed = {m.get("name", "") for m in response.json().get("models", [])}
        # Ollama reports "qwen2.5:3b-instruct"; users often configure "qwen2.5:3b".
        if not any(name == self._model or name.startswith(f"{self._model}:") or
                   name.split(":")[0] == self._model.split(":")[0] for name in installed):
            return False, (
                f"Model {self._model!r} is not installed. Run: ollama pull {self._model}"
            )
        return True, "ok"

    def _payload(self, system: str, user: str, temperature: float | None, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            # Without this, Ollama evicts the model after a few minutes idle and
            # the next question pays a ~30s reload. Measured on an M1: first
            # call 32.3s, subsequent calls 2.2s, for identical work. num_ctx is
            # part of the cache key too, so it must not vary between requests.
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }

    def warm_up(self) -> None:
        """Load the model into memory so the first real question is not slow.

        Best-effort: a failure here is not worth blocking startup for, because
        the same failure will be reported properly by `available()` when the
        user actually asks something.
        """
        try:
            httpx.post(
                f"{self.host}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"num_ctx": self.num_ctx, "num_predict": 1},
                },
                timeout=self.timeout_seconds,
            )
            logger.info("Model %s warmed up and resident", self._model)
        except Exception as exc:  # noqa: BLE001
            logger.info("Model warm-up skipped (%s)", exc)

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json=self._payload(system, user, temperature, stream=False),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        content = (response.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise LLMError("Ollama returned an empty response.")
        return content

    def stream(self, system: str, user: str, *, temperature: float | None = None) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                f"{self.host}/api/chat",
                json=self._payload(system, user, temperature, stream=True),
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama stream failed: {exc}") from exc


class OpenAICompatibleProvider(LLMProvider):
    """For a local vLLM / LM Studio / llama.cpp server, or a hosted API.

    Opt-in only. Selecting it is an explicit statement that you are happy for
    document text to leave the machine.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout_seconds: int = 180,
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature if temperature is None else temperature,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return response.json()["choices"][0]["message"]["content"].strip()


def create_llm_provider(config) -> LLMProvider:
    provider = (config.provider or "ollama").lower()
    if provider == "ollama":
        return OllamaProvider(
            model=config.model, host=config.host, temperature=config.temperature,
            num_ctx=config.num_ctx, timeout_seconds=config.timeout_seconds,
            keep_alive=config.keep_alive,
        )
    if provider in ("openai", "openai_compatible"):
        if not config.base_url:
            raise ValueError("llm.base_url is required for the openai provider.")
        logger.warning(
            "Using a remote LLM provider: document text will leave this machine."
        )
        return OpenAICompatibleProvider(
            model=config.model, base_url=config.base_url, api_key=config.api_key,
            temperature=config.temperature, timeout_seconds=config.timeout_seconds,
        )
    raise ValueError(f"Unknown llm provider {provider!r}. Use 'ollama' or 'openai'.")
