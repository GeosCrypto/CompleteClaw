"""Abstract base classes for memory stores."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single item stored in memory."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Memory(abc.ABC):
    """Abstract base class for all memory implementations.

    A memory store persists conversation history (or any context) across
    agent steps or sessions.

    Example::

        class RedisMemory(Memory):
            def save(self, entry):
                ...
            def load(self):
                ...
            def clear(self):
                ...
    """

    @abc.abstractmethod
    def save(self, entry: MemoryEntry) -> None:
        """Persist a single :class:`MemoryEntry`."""

    @abc.abstractmethod
    def load(self) -> List[MemoryEntry]:
        """Return all stored :class:`MemoryEntry` objects."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Erase all stored entries."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def save_user(self, content: str, **metadata: Any) -> None:
        """Save a user-role message."""
        self.save(MemoryEntry(role="user", content=content, metadata=metadata))

    def save_assistant(self, content: str, **metadata: Any) -> None:
        """Save an assistant-role message."""
        self.save(MemoryEntry(role="assistant", content=content, metadata=metadata))

    def save_system(self, content: str, **metadata: Any) -> None:
        """Save a system-role message."""
        self.save(MemoryEntry(role="system", content=content, metadata=metadata))

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(entries={len(self)})"
