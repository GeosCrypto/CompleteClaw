"""Sliding-window buffer memory."""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from completeclaw.memory.base import Memory, MemoryEntry


class BufferMemory(Memory):
    """An in-memory sliding window of the most recent conversation entries.

    Parameters
    ----------
    max_entries:
        Maximum number of entries to keep.  Older entries are dropped
        when the limit is exceeded.  Use ``None`` for unlimited.

    Example::

        mem = BufferMemory(max_entries=20)
        mem.save_user("Hello")
        mem.save_assistant("Hi there!")
        print(len(mem))  # 2
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        self._store: deque[MemoryEntry] = deque(
            maxlen=max_entries
        )

    def save(self, entry: MemoryEntry) -> None:
        self._store.append(entry)

    def load(self) -> List[MemoryEntry]:
        return list(self._store)

    def clear(self) -> None:
        self._store.clear()
