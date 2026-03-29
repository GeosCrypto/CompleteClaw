"""Abstract base classes for tools/plugins."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result returned by a tool invocation."""

    output: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output


class Tool(abc.ABC):
    """Abstract base class for all tools/plugins.

    Subclass this and implement :meth:`run` to create a new tool.

    Class attributes
    ----------------
    name:
        Unique identifier used to call the tool (no spaces).
    description:
        Human-readable description shown to the agent/LLM.

    Example::

        class EchoTool(Tool):
            name = "echo"
            description = "Echoes whatever you pass as 'message'."

            def run(self, message: str = "", **kwargs) -> ToolResult:
                return ToolResult(output=message)
    """

    name: str = "base_tool"
    description: str = "A tool."

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a :class:`ToolResult`."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ToolRegistry:
    """Registry that holds a collection of tools and dispatches calls.

    Example::

        registry = ToolRegistry([CalculatorTool(), EchoTool()])
        result = registry.call("calculator", expression="2 + 2")
        print(result.output)   # "4"
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self.tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add *tool* to the registry."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool with *name* or ``None`` if not found."""
        return self.tools.get(name)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        """Dispatch a call to the named tool."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{name}' not found in registry.",
            )
        return tool.run(**kwargs)

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self.tools.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()!r})"
