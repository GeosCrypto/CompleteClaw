"""Abstract base classes for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for API calls."""
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Response returned by an LLM provider."""

    content: str
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers.

    Subclass this to add support for a new LLM backend.

    Example::

        class MyProvider(LLMProvider):
            def chat(self, messages, **kwargs):
                ...
                return LLMResponse(content="hello")
    """

    name: str = "base"

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send *messages* to the LLM and return the response."""

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Simple text-in, text-out convenience wrapper around :meth:`chat`."""
        messages: List[Message] = []
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        messages.append(Message(role=Role.USER, content=prompt))
        return self.chat(messages, **kwargs).content

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens from the LLM.

        The default implementation calls :meth:`chat` and yields the full
        response as a single chunk.  Subclasses should override this method
        to provide true token-by-token streaming.
        """
        response = self.chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs
        )
        yield response.content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
