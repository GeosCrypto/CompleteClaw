"""LLM-compressed summary memory."""

from __future__ import annotations

from typing import Any, List

from completeclaw.memory.base import Memory, MemoryEntry

_SUMMARISE_PROMPT = """\
Summarise the following conversation history in a few concise sentences,
preserving key facts and decisions:

{history}

Summary:"""


class SummaryMemory(Memory):
    """Memory that compresses old entries into a running summary.

    When *max_entries* is exceeded, older entries are summarised by the
    LLM and replaced with a single summary entry, keeping the total
    entry count bounded.

    Parameters
    ----------
    llm:
        LLM provider used to generate summaries.
    max_entries:
        Number of entries before triggering summarisation.

    Example::

        from completeclaw.llm.mock import MockLLMProvider
        mem = SummaryMemory(llm=MockLLMProvider(responses=["Summary: …"]),
                            max_entries=4)
    """

    def __init__(self, llm: Any, *, max_entries: int = 10) -> None:
        from completeclaw.llm.base import LLMProvider

        self._llm: LLMProvider = llm
        self._max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def save(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._compress()

    def load(self) -> List[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def _compress(self) -> None:
        """Summarise the oldest half of stored entries."""
        half = len(self._entries) // 2
        to_compress = self._entries[:half]
        self._entries = self._entries[half:]

        history = "\n".join(f"{e.role}: {e.content}" for e in to_compress)
        summary_text = self._llm.complete(
            _SUMMARISE_PROMPT.format(history=history)
        )
        summary_entry = MemoryEntry(role="system", content=f"[Summary] {summary_text}")
        self._entries.insert(0, summary_entry)
