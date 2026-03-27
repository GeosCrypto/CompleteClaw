"""Buffer memory – keeps a sliding window of recent messages."""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from completeclaw.memory.base import Memory, MemoryEntry


class BufferMemory(Memory):
    """In-memory buffer that retains the *k* most recent messages.

    Parameters
    ----------
    max_entries:
        Maximum number of entries to keep.  Older entries are dropped
        automatically.  Pass ``None`` (default) for unlimited storage.

    Example::

        mem = BufferMemory(max_entries=10)
        mem.save_user("Hello")
        mem.save_assistant("Hi there!")
        print(len(mem))   # 2
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        self._max = max_entries
        self._entries: deque = deque(maxlen=max_entries)

    def save(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def load(self) -> List[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def __repr__(self) -> str:
        return f"BufferMemory(entries={len(self._entries)}, max={self._max!r})"
