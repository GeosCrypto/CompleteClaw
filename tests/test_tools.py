"""Tests for tools."""

from __future__ import annotations

import os
import tempfile

import pytest

from completeclaw.tools.base import Tool, ToolRegistry, ToolResult
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileReadTool, FileWriteTool
from completeclaw.tools.search import WebSearchTool

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_str(self):
        r = ToolResult(output="hello")
        assert str(r) == "hello"

    def test_defaults(self):
        r = ToolResult(output="x")
        assert r.success is True
        assert r.error is None


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    name = "echo"
    description = "Echoes message."

    def run(self, *, message: str = "", **kwargs) -> ToolResult:
        return ToolResult(output=message)


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry([EchoTool()])
        assert reg.get("echo") is not None

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("missing") is None

    def test_call_dispatches(self):
        reg = ToolRegistry([EchoTool()])
        result = reg.call("echo", message="hi")
        assert result.output == "hi"

    def test_call_missing_tool(self):
        reg = ToolRegistry()
        result = reg.call("missing")
        assert result.success is False

    def test_list_tools(self):
        reg = ToolRegistry([EchoTool()])
        assert "echo" in reg.list_tools()

    def test_repr(self):
        reg = ToolRegistry([EchoTool()])
        assert "echo" in repr(reg)


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def test_simple_addition(self):
        r = CalculatorTool().run(expression="2 + 2")
        assert r.output == "4"
        assert r.success

    def test_multiplication(self):
        r = CalculatorTool().run(expression="6 * 7")
        assert r.output == "42"

    def test_sqrt(self):
        r = CalculatorTool().run(expression="sqrt(144)")
        assert r.output == "12"

    def test_float_result(self):
        r = CalculatorTool().run(expression="1 / 3")
        assert "0.3333" in r.output

    def test_power(self):
        r = CalculatorTool().run(expression="2 ** 10")
        assert r.output == "1024"

    def test_pi_constant(self):
        r = CalculatorTool().run(expression="pi")
        assert float(r.output) == pytest.approx(3.14159, rel=1e-4)

    def test_invalid_expression(self):
        r = CalculatorTool().run(expression="import os")
        assert r.success is False

    def test_division_by_zero(self):
        r = CalculatorTool().run(expression="1 / 0")
        assert r.success is False


# ---------------------------------------------------------------------------
# FileReadTool / FileWriteTool
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            path = f.name
        try:
            write_result = FileWriteTool().run(path=path, content="hello world")
            assert write_result.success
            read_result = FileReadTool().run(path=path)
            assert read_result.output == "hello world"
        finally:
            os.unlink(path)

    def test_read_missing_file(self):
        r = FileReadTool().run(path="/nonexistent/path/file.txt")
        assert r.success is False

    def test_write_creates_directories(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "subdir", "file.txt")
            r = FileWriteTool().run(path=path, content="data")
            assert r.success
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# WebSearchTool (stub)
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def test_returns_stub_result(self):
        r = WebSearchTool().run(query="Python AI")
        assert r.success
        assert "Python AI" in r.output
