"""
CompleteClaw – The AI Swiss Army Knife.

A modular, extensible, and future-proof framework for all AI workflows,
from local LLMs to enterprise agents.

Quick start::

    from completeclaw import MockLLMProvider, ReActAgent, CalculatorTool

    llm = MockLLMProvider(responses=["Final Answer: 42"])
    agent = ReActAgent(llm, tools=[CalculatorTool()])
    result = agent.run("What is 6 * 7?")
    print(result.output)  # "42"
"""

# Agents
from completeclaw.agents import Agent, AgentResult, PlanAndExecuteAgent, ReActAgent

# LLM providers
from completeclaw.llm import LLMProvider, LLMResponse, Message, MockLLMProvider, Role

# Memory
from completeclaw.memory import BufferMemory, Memory, MemoryEntry, SummaryMemory

# Tools
from completeclaw.tools import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    HttpRequestTool,
    Tool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)

# Utilities
from completeclaw.utils import get_logger
from completeclaw.version import __version__

# Workflows
from completeclaw.workflows import (
    Pipeline,
    SequentialChain,
    Workflow,
    WorkflowResult,
    WorkflowStep,
)

__all__ = [
    "__version__",
    # LLM
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MockLLMProvider",
    "Role",
    # Agents
    "Agent",
    "AgentResult",
    "PlanAndExecuteAgent",
    "ReActAgent",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "FileReadTool",
    "FileWriteTool",
    "HttpRequestTool",
    "WebSearchTool",
    # Memory
    "Memory",
    "MemoryEntry",
    "BufferMemory",
    "SummaryMemory",
    # Workflows
    "Workflow",
    "WorkflowResult",
    "WorkflowStep",
    "SequentialChain",
    "Pipeline",
    # Utils
    "get_logger",
]
