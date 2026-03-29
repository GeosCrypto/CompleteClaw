"""OpenAI LLM provider."""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Sequence

from completeclaw.llm.base import LLMProvider, LLMResponse, Message


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API.

    Requires the ``openai`` package (``pip install completeclaw[openai]``).

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when omitted.
    default_model:
        Model to use when none is specified per-call.
    base_url:
        Optional base URL for Azure OpenAI or custom proxies.

    Example::

        llm = OpenAIProvider(default_model="gpt-4o-mini")
        print(llm.complete("Hello!"))
    """

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        default_model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires 'openai'. Install with: pip install completeclaw[openai]"
            ) from exc

        self._openai = _openai
        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
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
        api_messages = [m.to_dict() for m in messages]
        params: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        completion = self._client.chat.completions.create(**params)
        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        return LLMResponse(
            content=content,
            model=completion.model,
            usage=usage,
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
        api_messages = [m.to_dict() for m in messages]
        params: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": api_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
