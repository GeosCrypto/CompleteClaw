"""File I/O tools – read and write local files."""

from __future__ import annotations

import os
from typing import Any

from completeclaw.tools.base import Tool, ToolResult


class FileReadTool(Tool):
    """Read the contents of a local file.

    Parameters
    ----------
    path:
        Absolute or relative path to the file to read.
    encoding:
        Text encoding (default ``utf-8``).

    Example::

        tool = FileReadTool()
        result = tool.run(path="/tmp/hello.txt")
        print(result.output)
    """

    name = "file_read"
    description = (
        "Read the contents of a local file.  "
        "Input: {\"path\": \"<file path>\"}."
    )

    def run(self, *, path: str = "", encoding: str = "utf-8", **kwargs: Any) -> ToolResult:  # type: ignore[override]
        if not path:
            return ToolResult(output="", success=False, error="No path provided.")
        try:
            with open(path, encoding=encoding) as fh:
                content = fh.read()
            return ToolResult(output=content, metadata={"path": path, "bytes": len(content)})
        except OSError as exc:
            return ToolResult(output="", success=False, error=str(exc))


class FileWriteTool(Tool):
    """Write text content to a local file.

    Parameters
    ----------
    path:
        Destination file path.
    content:
        Text to write.
    mode:
        File open mode: ``"w"`` (overwrite, default) or ``"a"`` (append).
    encoding:
        Text encoding (default ``utf-8``).

    Example::

        tool = FileWriteTool()
        result = tool.run(path="/tmp/out.txt", content="Hello, world!")
        print(result.output)   # "Written 13 bytes to /tmp/out.txt"
    """

    name = "file_write"
    description = (
        "Write text content to a local file.  "
        "Input: {\"path\": \"<file path>\", \"content\": \"<text>\"}."
    )

    def run(  # type: ignore[override]
        self,
        *,
        path: str = "",
        content: str = "",
        mode: str = "w",
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(output="", success=False, error="No path provided.")
        if mode not in {"w", "a"}:
            return ToolResult(output="", success=False,
                              error=f"Invalid mode '{mode}'. Use 'w' or 'a'.")
        try:
            # Ensure parent directories exist
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, mode=mode, encoding=encoding) as fh:
                fh.write(content)
            msg = f"Written {len(content)} bytes to {path}"
            return ToolResult(output=msg, metadata={"path": path, "bytes": len(content)})
        except OSError as exc:
            return ToolResult(output="", success=False, error=str(exc))
