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
from completeclaw.agents import (
    Agent,
    AgentResult,
    PlanAndExecuteAgent,
    ReActAgent,
    RouterAgent,
)

# LLM providers
from completeclaw.llm import LLMProvider, LLMResponse, Message, MockLLMProvider, Role

# Memory
from completeclaw.memory import BufferMemory, FileMemory, Memory, MemoryEntry, SummaryMemory

# Tools
from completeclaw.tools import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    HttpRequestTool,
    JsonTool,
    Tool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)

# Utilities
from completeclaw.utils import get_logger, retry
from completeclaw.version import __version__

# Workflows
from completeclaw.workflows import (
    BranchStep,
    ConditionalChain,
    ConditionalStep,
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
    "RouterAgent",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "FileReadTool",
    "FileWriteTool",
    "HttpRequestTool",
    "JsonTool",
    "WebSearchTool",
    # Memory
    "Memory",
    "MemoryEntry",
    "BufferMemory",
    "FileMemory",
    "SummaryMemory",
    # Workflows
    "Workflow",
    "WorkflowResult",
    "WorkflowStep",
    "BranchStep",
    "ConditionalChain",
    "ConditionalStep",
    "SequentialChain",
    "Pipeline",
    # Utils
    "get_logger",
    "retry",
]
