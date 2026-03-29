"""Tests for memory systems."""

from __future__ import annotations

from completeclaw.llm.mock import MockLLMProvider
from completeclaw.memory.base import MemoryEntry
from completeclaw.memory.buffer import BufferMemory
from completeclaw.memory.summary import SummaryMemory


class TestMemoryEntry:
    def test_fields(self):
        e = MemoryEntry(role="user", content="hello")
        assert e.role == "user"
        assert e.content == "hello"


class TestBufferMemory:
    def test_save_and_load(self):
        mem = BufferMemory()
        mem.save_user("hello")
        mem.save_assistant("world")
        entries = mem.load()
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].role == "assistant"

    def test_clear(self):
        mem = BufferMemory()
        mem.save_user("x")
        mem.clear()
        assert len(mem.load()) == 0

    def test_max_entries_respected(self):
        mem = BufferMemory(max_entries=3)
        for i in range(5):
            mem.save_user(f"msg {i}")
        assert len(mem.load()) == 3

    def test_len(self):
        mem = BufferMemory()
        mem.save_user("a")
        assert len(mem) == 1

    def test_repr(self):
        mem = BufferMemory()
        assert "BufferMemory" in repr(mem)

    def test_save_system(self):
        mem = BufferMemory()
        mem.save_system("instructions")
        entries = mem.load()
        assert entries[0].role == "system"


class TestSummaryMemory:
    def test_basic_save_and_load(self):
        llm = MockLLMProvider(responses=["Summary of conversation"])
        mem = SummaryMemory(llm=llm, max_entries=4)
        mem.save_user("hi")
        mem.save_assistant("hello")
        entries = mem.load()
        assert len(entries) >= 1

    def test_compression_triggered(self):
        llm = MockLLMProvider(responses=["Summary: ..."] * 10)
        mem = SummaryMemory(llm=llm, max_entries=4)
        for i in range(6):
            mem.save_user(f"message {i}")
        entries = mem.load()
        # After compression, should have fewer entries than 6
        assert len(entries) < 6

    def test_clear(self):
        llm = MockLLMProvider()
        mem = SummaryMemory(llm=llm)
        mem.save_user("x")
        mem.clear()
        assert len(mem.load()) == 0
