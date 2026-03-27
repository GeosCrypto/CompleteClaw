"""Tool/plugin system."""

from completeclaw.tools.base import Tool, ToolResult, ToolRegistry
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileReadTool, FileWriteTool
from completeclaw.tools.search import WebSearchTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "FileReadTool",
    "FileWriteTool",
    "WebSearchTool",
]
