"""Ollama LLM provider for local models."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence
from urllib.request import Request, urlopen

from completeclaw.llm.base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """LLM provider that talks to a local Ollama server.

    No API key needed – requires a running ``ollama serve`` instance.

    Parameters
    ----------
    base_url:
        URL of the Ollama REST API.
    default_model:
        Ollama model tag to use when none is specified per-call.

    Example::

        llm = OllamaProvider(default_model="llama3")
        print(llm.complete("Hello!"))
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        default_model: str = "llama3",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data = json.dumps(payload).encode()
        req = Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as resp:
            body = json.loads(resp.read())

        content = body.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=body.get("model", model or self.default_model),
            usage={
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "completion_tokens": body.get("eval_count", 0),
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
        payload = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data = json.dumps(payload).encode()
        req = Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as resp:
            for line in resp:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
