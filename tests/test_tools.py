"""Tests for the Tool base classes and built-in tools."""

from __future__ import annotations

import os
import tempfile

from completeclaw.tools.base import ToolRegistry, ToolResult
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileReadTool, FileWriteTool
from completeclaw.tools.search import WebSearchTool

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_str(self):
        r = ToolResult(output="done")
        assert str(r) == "done"

    def test_success_default(self):
        r = ToolResult(output="x")
        assert r.success is True

    def test_failure(self):
        r = ToolResult(output="", success=False, error="oops")
        assert not r.success
        assert r.error == "oops"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def _make_registry(self):
        return ToolRegistry([CalculatorTool(), FileReadTool()])

    def test_register_and_get(self):
        reg = self._make_registry()
        assert reg.get("calculator") is not None
        assert reg.get("file_read") is not None
        assert reg.get("nonexistent") is None

    def test_call_dispatches(self):
        reg = self._make_registry()
        result = reg.call("calculator", expression="2 + 2")
        assert result.success
        assert result.output == "4"

    def test_call_unknown_tool(self):
        reg = self._make_registry()
        result = reg.call("unknown_tool")
        assert not result.success
        assert "not found" in result.error

    def test_list_tools(self):
        reg = self._make_registry()
        names = reg.list_tools()
        assert "calculator" in names

    def test_register_at_runtime(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        assert "calculator" in reg.list_tools()


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_addition(self):
        r = self.tool.run(expression="1 + 1")
        assert r.success
        assert r.output == "2"

    def test_subtraction(self):
        r = self.tool.run(expression="10 - 3")
        assert r.output == "7"

    def test_multiplication(self):
        r = self.tool.run(expression="6 * 7")
        assert r.output == "42"

    def test_division(self):
        r = self.tool.run(expression="10 / 4")
        assert r.output == "2.5"

    def test_exponentiation(self):
        r = self.tool.run(expression="2 ** 10")
        assert r.output == "1024"

    def test_floor_division(self):
        r = self.tool.run(expression="10 // 3")
        assert r.output == "3"

    def test_modulo(self):
        r = self.tool.run(expression="10 % 3")
        assert r.output == "1"

    def test_sqrt(self):
        r = self.tool.run(expression="sqrt(144)")
        assert r.output == "12"

    def test_pi_constant(self):
        import math
        r = self.tool.run(expression="pi")
        assert r.success
        assert abs(float(r.output) - math.pi) < 1e-6

    def test_complex_expression(self):
        r = self.tool.run(expression="sqrt(9) + 2 ** 3")
        assert r.output == "11"

    def test_empty_expression(self):
        r = self.tool.run(expression="")
        assert not r.success
        assert "No expression provided" in r.error

    def test_disallowed_builtin(self):
        r = self.tool.run(expression="__import__('os').getcwd()")
        assert not r.success

    def test_disallowed_name(self):
        r = self.tool.run(expression="open('secret')")
        assert not r.success

    def test_unary_minus(self):
        r = self.tool.run(expression="-5")
        assert r.output == "-5"

    def test_nested_function_calls(self):
        r = self.tool.run(expression="sqrt(abs(-16))")
        assert r.output == "4"


# ---------------------------------------------------------------------------
# FileReadTool / FileWriteTool
# ---------------------------------------------------------------------------


class TestFileTools:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hello.txt")
            writer = FileWriteTool()
            r = writer.run(path=path, content="Hello, CompleteClaw!")
            assert r.success, r.error

            reader = FileReadTool()
            r2 = reader.run(path=path)
            assert r2.success, r2.error
            assert r2.output == "Hello, CompleteClaw!"

    def test_write_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "append.txt")
            writer = FileWriteTool()
            writer.run(path=path, content="line1\n")
            writer.run(path=path, content="line2\n", mode="a")
            reader = FileReadTool()
            r = reader.run(path=path)
            assert "line1" in r.output and "line2" in r.output

    def test_read_nonexistent(self):
        reader = FileReadTool()
        r = reader.run(path="/nonexistent/path/file.txt")
        assert not r.success
        assert r.error

    def test_write_no_path(self):
        writer = FileWriteTool()
        r = writer.run(path="", content="x")
        assert not r.success

    def test_write_invalid_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = FileWriteTool()
            r = writer.run(path=os.path.join(tmpdir, "f.txt"), content="x", mode="z")
            assert not r.success

    def test_read_no_path(self):
        reader = FileReadTool()
        r = reader.run(path="")
        assert not r.success

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c.txt")
            writer = FileWriteTool()
            r = writer.run(path=path, content="nested")
            assert r.success
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# WebSearchTool (stub)
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def test_stub_returns_no_results(self):
        tool = WebSearchTool()
        r = tool.run(query="test query")
        assert r.success
        assert "No results found" in r.output

    def test_no_query(self):
        tool = WebSearchTool()
        r = tool.run(query="")
        assert not r.success

    def test_custom_subclass(self):
        class FakeSearch(WebSearchTool):
            def _search(self, query, num_results):
                return [{"title": "T", "url": "http://x.com", "snippet": "S"}]

        tool = FakeSearch()
        r = tool.run(query="anything")
        assert r.success
        assert "T" in r.output
        assert "http://x.com" in r.output
