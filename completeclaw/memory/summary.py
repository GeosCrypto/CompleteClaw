"""Summary memory – condenses old messages via an LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from completeclaw.memory.base import Memory, MemoryEntry

if TYPE_CHECKING:
    from completeclaw.llm.base import LLMProvider


_SUMMARIZE_PROMPT = """\
Summarize the following conversation into a concise paragraph that preserves
the key context and decisions:

{history}

Summary:"""


class SummaryMemory(Memory):
    """Memory that keeps a running summary plus the most recent *k* turns.

    When the number of entries exceeds *max_before_summary*, an LLM is used
    to compress the oldest half of the buffer into a single summary entry.

    Parameters
    ----------
    llm:
        LLM provider used for summarization.
    max_before_summary:
        Trigger summarization once this many entries accumulate.
    recent_k:
        Number of most-recent entries to keep verbatim after summarization.

    Example::

        from completeclaw.llm.mock import MockLLMProvider
        llm = MockLLMProvider(responses=["A concise summary."])
        mem = SummaryMemory(llm=llm, max_before_summary=6)
    """

    def __init__(
        self,
        llm: "LLMProvider",
        *,
        max_before_summary: int = 20,
        recent_k: int = 4,
    ) -> None:
        self._llm = llm
        self._max = max_before_summary
        self._recent_k = recent_k
        self._entries: List[MemoryEntry] = []
        self._summary: Optional[str] = None

    def save(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) >= self._max:
            self._compress()

    def load(self) -> List[MemoryEntry]:
        result: List[MemoryEntry] = []
        if self._summary:
            result.append(MemoryEntry(role="system", content=f"Summary: {self._summary}"))
        result.extend(self._entries)
        return result

    def clear(self) -> None:
        self._entries.clear()
        self._summary = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        """Summarize old entries and retain only the most recent ones."""
        to_summarize = self._entries[: -self._recent_k]
        keep = self._entries[-self._recent_k :]

        history_text = "\n".join(
            f"{e.role.upper()}: {e.content}" for e in to_summarize
        )
        prompt = _SUMMARIZE_PROMPT.format(history=history_text)
        new_summary = self._llm.complete(prompt)

        if self._summary:
            new_summary = f"{self._summary} {new_summary}"

        self._summary = new_summary
        self._entries = keep

    def __repr__(self) -> str:
        return (
            f"SummaryMemory("
            f"entries={len(self._entries)}, "
            f"has_summary={self._summary is not None!r})"
        )
