"""Tool/plugin system."""

from completeclaw.tools.base import Tool, ToolRegistry, ToolResult
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileReadTool, FileWriteTool
from completeclaw.tools.http import HttpRequestTool
from completeclaw.tools.json_tool import JsonTool
from completeclaw.tools.search import WebSearchTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "FileReadTool",
    "FileWriteTool",
    "HttpRequestTool",
    "JsonTool",
    "WebSearchTool",
]
