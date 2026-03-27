"""Ollama local LLM provider."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Iterator, Optional, Sequence

from completeclaw.llm.base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance.

    No extra Python packages are required – communication happens over
    Ollama's built-in HTTP API using only the standard library.

    Parameters
    ----------
    host:
        Base URL of the Ollama server.  Defaults to ``http://localhost:11434``.
    default_model:
        Model to pull/use when *model* is not provided to :meth:`chat`.

    Example::

        from completeclaw.llm.ollama import OllamaProvider

        llm = OllamaProvider(default_model="llama3")
        reply = llm.complete("Tell me a joke.")
        print(reply)
    """

    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        default_model: str = "llama3",
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._host}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            return json.loads(resp.read())

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict = {
            "model": model or self._default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        result = self._post("/api/chat", payload)
        content: str = result.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=result.get("model", model or self._default_model),
            usage={
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
            },
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        import urllib.request

        url = f"{self._host}/api/chat"
        payload: dict = {
            "model": model or self._default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    def list_models(self) -> list:
        """Return a list of locally available model names."""
        url = f"{self._host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
