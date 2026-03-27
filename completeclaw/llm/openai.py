"""OpenAI LLM provider."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence

from completeclaw.llm.base import LLMProvider, LLMResponse, Message, Role


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API.

    Requires the ``openai`` Python package (``pip install completeclaw[openai]``).

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not supplied.
    default_model:
        Model to use when *model* is not passed to :meth:`chat`.
    base_url:
        Override the API base URL (useful for Azure OpenAI or proxies).

    Example::

        from completeclaw.llm.openai import OpenAIProvider

        llm = OpenAIProvider(api_key="sk-...")
        reply = llm.complete("What is 2 + 2?")
        print(reply)   # "4"
    """

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        default_model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required.  "
                "Install it with: pip install completeclaw[openai]"
            ) from exc

        import os

        from openai import OpenAI  # type: ignore[import]

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            **client_kwargs,
        )
        self._default_model = default_model

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
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
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
        payload: dict = {
            "model": model or self._default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        for chunk in self._client.chat.completions.create(**payload):
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
