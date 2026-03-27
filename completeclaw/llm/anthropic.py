"""Anthropic LLM provider."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence

from completeclaw.llm.base import LLMProvider, LLMResponse, Message, Role


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API.

    Requires the ``anthropic`` Python package
    (``pip install completeclaw[anthropic]``).

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when not supplied.
    default_model:
        Model to use when *model* is not passed to :meth:`chat`.

    Example::

        from completeclaw.llm.anthropic import AnthropicProvider

        llm = AnthropicProvider(api_key="sk-ant-...")
        reply = llm.complete("What is 2 + 2?")
        print(reply)   # "4"
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        default_model: str = "claude-3-5-haiku-latest",
        **client_kwargs: Any,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required.  "
                "Install it with: pip install completeclaw[anthropic]"
            ) from exc

        import os

        from anthropic import Anthropic  # type: ignore[import]

        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
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
        system_prompt = ""
        user_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                user_messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict = {
            "model": model or self._default_model,
            "messages": user_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)

        response = self._client.messages.create(**payload)
        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content=content, model=response.model, usage=usage)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        system_prompt = ""
        user_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                user_messages.append({"role": msg.role.value, "content": msg.content})

        payload: dict = {
            "model": model or self._default_model,
            "messages": user_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)

        with self._client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                yield text
