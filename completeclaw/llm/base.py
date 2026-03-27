"""Abstract base classes for LLM providers."""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence


class Role(str, enum.Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single chat message."""

    role: Role
    content: str
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Response returned by an LLM provider."""

    content: str
    model: str
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers.

    Subclass this to integrate any LLM backend (OpenAI, Anthropic, Ollama,
    local models, custom endpoints, …).

    Example::

        class MyProvider(LLMProvider):
            def chat(self, messages, **kwargs):
                ...
    """

    name: str = "base"

    # ------------------------------------------------------------------
    # Core interface – must be implemented by subclasses
    # ------------------------------------------------------------------

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
        """Send a list of messages and return a single response."""

    # ------------------------------------------------------------------
    # Optional streaming interface
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield response tokens one by one.

        Default implementation calls :meth:`chat` and yields the whole
        response as a single chunk.  Override for true streaming support.
        """
        response = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        yield response.content

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """High-level helper: send a plain-text prompt and return the reply."""
        messages: List[Message] = []
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        messages.append(Message(role=Role.USER, content=prompt))
        return self.chat(messages, **kwargs).content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
