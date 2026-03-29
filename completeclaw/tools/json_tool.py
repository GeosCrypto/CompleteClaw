"""JSON processing tool."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from completeclaw.tools.base import Tool, ToolResult


class JsonTool(Tool):
    """Parse, query, and format JSON data using only the standard library.

    Supports three operations via the ``operation`` parameter:

    * ``"parse"`` – deserialise a JSON string and return pretty-printed output.
    * ``"query"`` – extract a nested value using a dot-notation key path
      (e.g. ``"user.address.city"`` or ``"results.0.title"`` for list access).
    * ``"format"`` – round-trip a JSON string through the parser and return
      human-readable indented output (key ordering is *not* guaranteed, but
      whitespace is standardised).

    Parameters in ``run()``
    -----------------------
    operation:
        One of ``"parse"``, ``"query"``, or ``"format"``.  Defaults to
        ``"parse"``.
    data:
        The JSON string to operate on.
    path:
        Dot-notation key path for the ``"query"`` operation.
        Integer parts index into lists, e.g. ``"items.0.name"``.
    indent:
        Indentation level used for pretty-printing.  Defaults to ``2``.

    Example::

        tool = JsonTool()

        # Parse / pretty-print
        r = tool.run(data='{"name":"Alice","age":30}')
        print(r.output)
        # {
        #   "name": "Alice",
        #   "age": 30
        # }

        # Query a nested value
        r = tool.run(operation="query", data='{"user":{"name":"Bob"}}', path="user.name")
        print(r.output)  # "Bob"

        # Query a list element
        r = tool.run(operation="query", data='{"items":[10,20,30]}', path="items.1")
        print(r.output)  # "20"
    """

    name = "json_tool"
    description = (
        "Parse, query, or format JSON. "
        "Input: {'operation': 'parse|query|format', 'data': '<json string>', "
        "'path': '<dot.notation.path> (query only)', 'indent': 2}."
    )

    def run(
        self,
        *,
        operation: str = "parse",
        data: str = "",
        path: Optional[str] = None,
        indent: int = 2,
        **kwargs: Any,
    ) -> ToolResult:
        if not data:
            return ToolResult(output="", success=False, error="'data' is required.")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            return ToolResult(output="", success=False, error=f"Invalid JSON: {exc}")

        op = operation.lower().strip()

        if op in ("parse", "format"):
            return ToolResult(
                output=json.dumps(parsed, indent=indent, ensure_ascii=False),
                metadata={"operation": op, "type": type(parsed).__name__},
            )

        if op == "query":
            if not path:
                return ToolResult(
                    output="", success=False, error="'path' is required for query."
                )
            value, error = _dot_get(parsed, path)
            if error:
                return ToolResult(output="", success=False, error=error)
            if isinstance(value, (dict, list)):
                out = json.dumps(value, indent=indent, ensure_ascii=False)
            else:
                out = str(value) if value is not None else "null"
            return ToolResult(output=out, metadata={"operation": op, "path": path})

        return ToolResult(
            output="",
            success=False,
            error=f"Unknown operation '{operation}'. Use 'parse', 'query', or 'format'.",
        )


def _dot_get(obj: Any, path: str) -> Tuple[Any, Optional[str]]:
    """Navigate *obj* via a dot-separated path, supporting integer list indices."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None, f"Key '{part}' not found."
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return None, f"Expected an integer index for list, got '{part}'."
            if idx >= len(current) or idx < -len(current):
                return None, f"List index {idx} out of range (length {len(current)})."
            current = current[idx]
        else:
            return None, f"Cannot navigate into {type(current).__name__} at '{part}'."
    return current, None
