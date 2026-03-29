"""Tests for tools."""

from __future__ import annotations

import http.server
import os
import tempfile
import threading

import pytest

from completeclaw.tools.base import Tool, ToolRegistry, ToolResult
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileReadTool, FileWriteTool
from completeclaw.tools.http import HttpRequestTool
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


# ---------------------------------------------------------------------------
# HttpRequestTool
# ---------------------------------------------------------------------------


class _SimpleHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler used in tests – no logging to stdout."""

    def log_message(self, *args):  # silence server log output
        pass

    def do_GET(self):
        if self.path == "/ok":
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/error":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)  # echo the body back


def _start_server():
    """Start a one-shot local HTTP server and return (server, base_url)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _SimpleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


class TestHttpRequestTool:
    @classmethod
    def setup_class(cls):
        cls.server, cls.base_url = _start_server()

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()

    def test_get_request_success(self):
        tool = HttpRequestTool()
        result = tool.run(url=f"{self.base_url}/ok")
        assert result.success
        assert "ok" in result.output
        assert result.metadata["status"] == 200

    def test_get_request_http_error(self):
        tool = HttpRequestTool()
        result = tool.run(url=f"{self.base_url}/error")
        assert result.success is False
        assert "404" in result.error

    def test_post_with_json_body(self):
        tool = HttpRequestTool()
        result = tool.run(url=f"{self.base_url}/post", method="POST", body={"key": "val"})
        assert result.success
        # Server echoes the body back
        assert "key" in result.output

    def test_missing_url_returns_error(self):
        tool = HttpRequestTool()
        result = tool.run(url="")
        assert result.success is False
        assert "url" in result.error.lower()

    def test_invalid_url_returns_error(self):
        tool = HttpRequestTool()
        result = tool.run(url="http://127.0.0.1:1/unreachable", timeout=1)
        assert result.success is False

    def test_tool_name_and_description(self):
        tool = HttpRequestTool()
        assert tool.name == "http_request"
        assert "HTTP" in tool.description

    def test_repr(self):
        assert "HttpRequestTool" in repr(HttpRequestTool())

