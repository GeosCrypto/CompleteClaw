"""Anthropic LLM provider."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from completeclaw.llm.base import LLMProvider, LLMResponse, Message, Role


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API (Claude models).

    Requires the ``anthropic`` package (``pip install completeclaw[anthropic]``).

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when omitted.
    default_model:
        Claude model to use when none is specified per-call.

    Example::

        llm = AnthropicProvider(default_model="claude-3-5-haiku-latest")
        print(llm.complete("Hello!"))
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        default_model: str = "claude-3-5-haiku-latest",
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires 'anthropic'. "
                "Install with: pip install completeclaw[anthropic]"
            ) from exc

        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.default_model = default_model

    def _split_messages(
        self, messages: Sequence[Message]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Separate the system prompt from the conversation messages."""
        system: Optional[str] = None
        api_messages: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system = m.content
            else:
                api_messages.append({"role": m.role.value, "content": m.content})
        return system, api_messages

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system, api_messages = self._split_messages(messages)
        params: Dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or 1024,
            "messages": api_messages,
            "temperature": temperature,
        }
        if system:
            params["system"] = system

        response = self._client.messages.create(**params)
        content = response.content[0].text if response.content else ""
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
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
        system, api_messages = self._split_messages(messages)
        params: Dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or 1024,
            "messages": api_messages,
            "temperature": temperature,
        }
        if system:
            params["system"] = system

        with self._client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
