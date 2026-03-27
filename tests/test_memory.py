"""Tests for memory implementations."""

from __future__ import annotations

import pytest

from completeclaw.memory.base import Memory, MemoryEntry
from completeclaw.memory.buffer import BufferMemory
from completeclaw.memory.summary import SummaryMemory
from completeclaw.llm.mock import MockLLMProvider


# ---------------------------------------------------------------------------
# BufferMemory
# ---------------------------------------------------------------------------


class TestBufferMemory:
    def test_save_and_load(self):
        mem = BufferMemory()
        mem.save_user("hello")
        mem.save_assistant("hi")
        entries = mem.load()
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].role == "assistant"

    def test_max_entries(self):
        mem = BufferMemory(max_entries=3)
        for i in range(5):
            mem.save_user(f"msg {i}")
        entries = mem.load()
        assert len(entries) == 3
        assert entries[0].content == "msg 2"
        assert entries[-1].content == "msg 4"

    def test_clear(self):
        mem = BufferMemory()
        mem.save_user("x")
        mem.clear()
        assert len(mem.load()) == 0

    def test_len(self):
        mem = BufferMemory()
        assert len(mem) == 0
        mem.save_user("a")
        assert len(mem) == 1

    def test_save_system(self):
        mem = BufferMemory()
        mem.save_system("You are a helpful assistant.")
        entries = mem.load()
        assert entries[0].role == "system"

    def test_unlimited(self):
        mem = BufferMemory()
        for i in range(100):
            mem.save_user(f"msg {i}")
        assert len(mem.load()) == 100

    def test_repr(self):
        mem = BufferMemory(max_entries=5)
        assert "BufferMemory" in repr(mem)


# ---------------------------------------------------------------------------
# SummaryMemory
# ---------------------------------------------------------------------------


class TestSummaryMemory:
    def _make_mem(self, max_before=6, recent_k=2):
        llm = MockLLMProvider(responses=["Summarized context."])
        return SummaryMemory(llm=llm, max_before_summary=max_before, recent_k=recent_k)

    def test_below_threshold_no_summary(self):
        mem = self._make_mem(max_before=10)
        for i in range(5):
            mem.save_user(f"msg {i}")
        entries = mem.load()
        assert all(e.role == "user" for e in entries)
        assert len(entries) == 5

    def test_compresses_when_threshold_reached(self):
        mem = self._make_mem(max_before=6, recent_k=2)
        for i in range(6):
            mem.save_user(f"msg {i}")
        entries = mem.load()
        # Should have a summary entry plus the recent_k user entries
        roles = [e.role for e in entries]
        assert "system" in roles  # summary injected as system message
        assert len([e for e in entries if e.role == "user"]) == 2

    def test_clear_removes_summary(self):
        mem = self._make_mem(max_before=6, recent_k=2)
        for i in range(6):
            mem.save_user(f"msg {i}")
        mem.clear()
        assert len(mem.load()) == 0

    def test_summary_content_present(self):
        mem = self._make_mem(max_before=6, recent_k=2)
        for i in range(6):
            mem.save_user(f"msg {i}")
        entries = mem.load()
        system_entries = [e for e in entries if e.role == "system"]
        assert any("Summarized context." in e.content for e in system_entries)
