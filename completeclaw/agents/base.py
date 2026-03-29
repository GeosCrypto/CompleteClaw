"""Abstract base classes for agents."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from completeclaw.llm.base import LLMProvider
from completeclaw.memory.base import Memory
from completeclaw.tools.base import Tool, ToolRegistry


@dataclass
class AgentResult:
    """Result returned by an agent after completing a task."""

    output: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output


class Agent(abc.ABC):
    """Abstract base class for all agents.

    An agent receives a goal/task and iteratively uses an LLM and
    (optionally) tools and memory to produce a result.

    Parameters
    ----------
    llm:
        The LLM provider the agent will use for reasoning.
    tools:
        Optional list of tools available to the agent.
    memory:
        Optional memory store for persisting context across steps.
    max_iterations:
        Hard cap on the number of reasoning/action steps before the
        agent gives up.
    system_prompt:
        Optional system-level instruction injected at the start of every
        conversation.

    Example::

        class MyAgent(Agent):
            def run(self, task, **kwargs):
                reply = self.llm.complete(task)
                return AgentResult(output=reply)
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = ToolRegistry(tools or [])
        self.memory = memory
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the given *task* and return an :class:`AgentResult`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_tool(self, tool: Tool) -> None:
        """Register an additional tool at runtime."""
        self.tool_registry.register(tool)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"llm={self.llm!r}, "
            f"tools={list(self.tool_registry.tools.keys())!r})"
        )
