"""File read/write tools."""

from __future__ import annotations

import os
from typing import Any

from completeclaw.tools.base import Tool, ToolResult


class FileReadTool(Tool):
    """Reads the contents of a file.

    Parameters
    ----------
    path:
        Absolute or relative path of the file to read.

    Example::

        tool = FileReadTool()
        result = tool.run(path="/tmp/hello.txt")
        print(result.output)
    """

    name = "file_read"
    description = (
        "Reads a file and returns its contents. "
        "Input: {'path': '<file path>'}."
    )

    def run(self, *, path: str = "", **kwargs: Any) -> ToolResult:
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            return ToolResult(output=content)
        except OSError as exc:
            return ToolResult(output="", success=False, error=str(exc))


class FileWriteTool(Tool):
    """Writes text to a file, creating it if necessary.

    Parameters
    ----------
    path:
        Destination file path.
    content:
        Text to write.

    Example::

        tool = FileWriteTool()
        result = tool.run(path="/tmp/out.txt", content="hello")
        print(result.output)  # "Written 5 bytes to /tmp/out.txt"
    """

    name = "file_write"
    description = (
        "Writes text to a file (creates or overwrites). "
        "Input: {'path': '<file path>', 'content': '<text to write>'}."
    )

    def run(self, *, path: str = "", content: str = "", **kwargs: Any) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return ToolResult(output=f"Written {len(content.encode())} bytes to {path}")
        except OSError as exc:
            return ToolResult(output="", success=False, error=str(exc))
