"""Mock LLM provider – useful for testing and offline development."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from completeclaw.llm.base import LLMProvider, LLMResponse, Message


class MockLLMProvider(LLMProvider):
    """An LLM provider that returns pre-defined or generated replies.

    Parameters
    ----------
    responses:
        Iterable of strings returned in order.  When exhausted the last
        response is repeated.
    reply_fn:
        Optional callable ``(messages) -> str`` for dynamic replies.
        Takes priority over *responses* when provided.
    model:
        Model name to report in responses.

    Example::

        llm = MockLLMProvider(responses=["Hello!", "How can I help?"])
        print(llm.complete("Hi"))   # "Hello!"
        print(llm.complete("?"))    # "How can I help?"
    """

    name = "mock"

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        *,
        reply_fn: Optional[Callable[[Sequence[Message]], str]] = None,
        model: str = "mock-model",
    ) -> None:
        self._responses = list(responses or ["I am a mock LLM response."])
        self._reply_fn = reply_fn
        self._model = model
        self._call_count = 0

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._reply_fn is not None:
            content = self._reply_fn(messages)
        else:
            idx = min(self._call_count, len(self._responses) - 1)
            content = self._responses[idx]

        self._call_count += 1

        return LLMResponse(
            content=content,
            model=model or self._model,
            usage={"prompt_tokens": 0, "completion_tokens": len(content.split())},
        )

    def reset(self) -> None:
        """Reset call counter (useful between test cases)."""
        self._call_count = 0
