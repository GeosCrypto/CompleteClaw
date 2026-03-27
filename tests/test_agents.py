"""Tests for the ReAct agent."""

from __future__ import annotations

from completeclaw.agents.react import ReActAgent
from completeclaw.llm.mock import MockLLMProvider
from completeclaw.memory.buffer import BufferMemory
from completeclaw.tools.base import Tool, ToolResult
from completeclaw.tools.calculator import CalculatorTool

# ---------------------------------------------------------------------------
# Helper tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    name = "echo"
    description = "Echoes the 'message' argument."

    def run(self, *, message: str = "", **kwargs) -> ToolResult:
        return ToolResult(output=message)


# ---------------------------------------------------------------------------
# ReActAgent tests
# ---------------------------------------------------------------------------


class TestReActAgent:
    def test_direct_final_answer(self):
        llm = MockLLMProvider(responses=["Final Answer: 42"])
        agent = ReActAgent(llm)
        result = agent.run("What is 6 * 7?")
        assert result.output == "42"
        assert result.metadata["iterations"] == 1

    def test_uses_tool_and_gets_final_answer(self):
        responses = [
            "Thought: I need to calculate.\nAction: calculator\nAction Input: {\"expression\": \"6 * 7\"}",
            "Final Answer: 42",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = ReActAgent(llm, tools=[CalculatorTool()])
        result = agent.run("What is 6 * 7?")
        assert result.output == "42"
        assert len(result.steps) == 1
        assert result.steps[0]["action"] == "calculator"
        assert result.steps[0]["observation"] == "42"

    def test_unknown_tool_returns_error_observation(self):
        responses = [
            "Action: nonexistent_tool\nAction Input: {}",
            "Final Answer: handled",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = ReActAgent(llm)
        result = agent.run("do something")
        assert result.output == "handled"
        assert "not found" in result.steps[0]["observation"]

    def test_max_iterations_exceeded(self):
        llm = MockLLMProvider(responses=["Action: echo\nAction Input: {\"message\": \"x\"}"])
        agent = ReActAgent(llm, tools=[EchoTool()], max_iterations=3)
        result = agent.run("loop forever")
        assert result.metadata.get("truncated") is True
        assert result.metadata["iterations"] == 3

    def test_no_action_returns_reply_directly(self):
        llm = MockLLMProvider(responses=["Just a plain reply without any action."])
        agent = ReActAgent(llm)
        result = agent.run("say something")
        assert result.output == "Just a plain reply without any action."

    def test_add_tool_at_runtime(self):
        responses = [
            "Action: echo\nAction Input: {\"message\": \"hi\"}",
            "Final Answer: done",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool())
        result = agent.run("test")
        assert result.steps[0]["observation"] == "hi"

    def test_memory_persists_conversation(self):
        mem = BufferMemory()
        llm = MockLLMProvider(responses=["Final Answer: remembered"])
        agent = ReActAgent(llm, memory=mem)
        agent.run("first task")
        entries = mem.load()
        assert any(e.role == "user" for e in entries)
        assert any(e.role == "assistant" for e in entries)

    def test_system_prompt_injected(self):
        captured = []

        def _reply(msgs):
            captured.extend(msgs)
            return "Final Answer: ok"

        llm = MockLLMProvider(reply_fn=_reply)
        agent = ReActAgent(llm, system_prompt="Be concise.")
        agent.run("task")
        system_msg = captured[0]
        assert "Be concise." in system_msg.content

    def test_repr(self):
        llm = MockLLMProvider()
        agent = ReActAgent(llm, tools=[CalculatorTool()])
        assert "ReActAgent" in repr(agent)
        assert "calculator" in repr(agent)
