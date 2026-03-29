"""Persistent file-backed memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from completeclaw.memory.base import Memory, MemoryEntry


class FileMemory(Memory):
    """Persistent memory that saves conversation entries to a JSON file.

    Every :meth:`save` and :meth:`clear` call immediately writes the updated
    entry list to disk, so no data is lost on process exit or crash.
    On construction the file is read (if it exists) and its entries are
    pre-loaded as the initial state.

    Parameters
    ----------
    path:
        File path for the backing JSON store.  Parent directories are
        created automatically.  If the file already exists its contents
        are loaded as the initial memory state.
    max_entries:
        Optional cap on the number of entries kept.  When the limit is
        exceeded the oldest entry is discarded (FIFO), maintaining a
        sliding window like :class:`~completeclaw.memory.buffer.BufferMemory`.

    Example::

        mem = FileMemory(path="/tmp/chat_history.json")
        mem.save_user("What is 2 + 2?")
        mem.save_assistant("4")

        # In a later process:
        mem2 = FileMemory(path="/tmp/chat_history.json")
        print(len(mem2))  # 2 – history loaded from disk
    """

    def __init__(
        self,
        path: str,
        *,
        max_entries: Optional[int] = None,
    ) -> None:
        self._path = Path(path)
        self._max_entries = max_entries
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[MemoryEntry] = self._load_from_disk()

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if self._max_entries is not None:
            while len(self._entries) > self._max_entries:
                self._entries.pop(0)
        self._flush()

    def load(self) -> List[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries = []
        self._flush()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> List[MemoryEntry]:
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return [MemoryEntry(**item) for item in data]
        except Exception:
            return []

    def _flush(self) -> None:
        data = [
            {"role": e.role, "content": e.content, "metadata": e.metadata}
            for e in self._entries
        ]
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def __repr__(self) -> str:
        return f"FileMemory(path={str(self._path)!r}, entries={len(self)})"
