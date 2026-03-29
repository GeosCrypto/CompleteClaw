"""Tests for memory systems."""

from __future__ import annotations

import json
import os
import tempfile

from completeclaw.llm.mock import MockLLMProvider
from completeclaw.memory.base import MemoryEntry
from completeclaw.memory.buffer import BufferMemory
from completeclaw.memory.file_memory import FileMemory
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


# ---------------------------------------------------------------------------
# FileMemory
# ---------------------------------------------------------------------------


class TestFileMemory:
    def _tmp_path(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(path)  # let FileMemory create it fresh
        return path

    def test_save_and_load(self):
        path = self._tmp_path()
        try:
            mem = FileMemory(path)
            mem.save_user("hello")
            mem.save_assistant("world")
            entries = mem.load()
            assert len(entries) == 2
            assert entries[0].role == "user"
            assert entries[1].role == "assistant"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistence_across_instances(self):
        path = self._tmp_path()
        try:
            mem1 = FileMemory(path)
            mem1.save_user("remember me")
            # A second instance loading from the same file should see the entry
            mem2 = FileMemory(path)
            entries = mem2.load()
            assert len(entries) == 1
            assert entries[0].content == "remember me"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_clear_empties_file(self):
        path = self._tmp_path()
        try:
            mem = FileMemory(path)
            mem.save_user("x")
            mem.clear()
            assert len(mem.load()) == 0
            # The file should contain an empty JSON array
            with open(path) as f:
                assert json.load(f) == []
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_max_entries_respected(self):
        path = self._tmp_path()
        try:
            mem = FileMemory(path, max_entries=3)
            for i in range(5):
                mem.save_user(f"msg {i}")
            assert len(mem.load()) == 3
            # Only the 3 most recent entries are kept
            assert mem.load()[0].content == "msg 2"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "subdir", "nested", "memory.json")
            mem = FileMemory(path)
            mem.save_user("hello")
            assert os.path.exists(path)

    def test_repr(self):
        path = self._tmp_path()
        try:
            mem = FileMemory(path)
            r = repr(mem)
            assert "FileMemory" in r
            assert "entries=0" in r
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_corrupt_file_loads_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{{")
            path = f.name
        try:
            mem = FileMemory(path)
            assert len(mem.load()) == 0
        finally:
            os.unlink(path)
