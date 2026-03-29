"""Tests for agents (ReActAgent and PlanAndExecuteAgent)."""

from __future__ import annotations

from completeclaw.agents.plan_execute import PlanAndExecuteAgent
from completeclaw.agents.react import ReActAgent
from completeclaw.agents.router import RouterAgent
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
            'Thought: I need to calculate.\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
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
        llm = MockLLMProvider(responses=['Action: echo\nAction Input: {"message": "x"}'])
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
            'Action: echo\nAction Input: {"message": "hi"}',
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


# ---------------------------------------------------------------------------
# PlanAndExecuteAgent tests
# ---------------------------------------------------------------------------


class TestPlanAndExecuteAgent:
    def test_basic_plan_and_execute(self):
        responses = [
            '["Step 1", "Step 2"]',   # planning response
            "Final Answer: done step 1",  # execute step 1
            "Final Answer: done step 2",  # execute step 2
            "Final Answer: all done",     # synthesis
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm)
        result = agent.run("do something complex")
        assert result.output == "all done"
        assert result.metadata["plan"] == ["Step 1", "Step 2"]
        assert result.metadata["plan_steps"] == 2
        assert len(result.steps) == 2

    def test_fallback_plan_when_json_invalid(self):
        responses = [
            "This is not valid JSON",  # planning response (fallback)
            "Final Answer: done",       # execute the single fallback step
            "Final Answer: synthesised",  # synthesis
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm)
        result = agent.run("my task")
        # Should fall back to using the task itself as the single step
        assert result.metadata["plan_steps"] == 1

    def test_max_plan_steps_limits_plan(self):
        big_plan = [f"Step {i}" for i in range(20)]
        import json
        responses = [
            json.dumps(big_plan),       # planning response with 20 steps
        ] + ["Final Answer: ok"] * 5 + ["Final Answer: synthesised"]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm, max_plan_steps=5)
        result = agent.run("big task")
        assert result.metadata["plan_steps"] == 5

    def test_step_uses_tool(self):
        responses = [
            '["Calculate 6*7"]',    # plan
            'Thought: need calc\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
            "Final Answer: 42",     # after tool observation
            "Final Answer: The answer is 42",  # synthesis
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm, tools=[CalculatorTool()])
        result = agent.run("What is 6 times 7?")
        assert "42" in result.output

    def test_memory_saves_conversation(self):
        mem = BufferMemory()
        responses = [
            '["Only step"]',
            "Final Answer: result",
            "Final Answer: synthesised",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm, memory=mem)
        agent.run("task with memory")
        entries = mem.load()
        assert any(e.role == "user" for e in entries)
        assert any(e.role == "assistant" for e in entries)

    def test_system_prompt_prepended_to_plan(self):
        captured = []

        def _reply(msgs):
            captured.extend(msgs)
            return '["step"]'

        llm = MockLLMProvider(reply_fn=_reply)
        agent = PlanAndExecuteAgent(llm, system_prompt="Be thorough.")
        # Just check the plan phase; execution will repeat the plan reply
        try:
            agent.run("task")
        except Exception:
            pass  # synthesis may fail with repeated plan reply
        # The first captured message should contain the system prompt
        assert any("Be thorough." in m.content for m in captured)

    def test_repr(self):
        llm = MockLLMProvider()
        agent = PlanAndExecuteAgent(llm, tools=[CalculatorTool()])
        assert "PlanAndExecuteAgent" in repr(agent)
        assert "calculator" in repr(agent)

    def test_step_max_iterations_fallback(self):
        responses = [
            '["Single step"]',  # plan (call 0)
            # execution: two action responses fill both iterations without a Final Answer
            'Action: echo\nAction Input: {"message": "loop"}',  # execute iter 0 (call 1)
            'Action: echo\nAction Input: {"message": "loop"}',  # execute iter 1 (call 2)
            # synthesis gets a valid response after the step exhausts max_iterations
            "Final Answer: synthesised despite loop",  # synthesis (call 3)
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm, tools=[EchoTool()], max_iterations=2)
        result = agent.run("loop task")
        # step result should be the "max iterations" fallback
        assert "Maximum iterations" in result.steps[0]["result"]

    def test_embedded_json_array_in_plan_response(self):
        responses = [
            'Here is your plan: ["Alpha", "Beta"] enjoy!',  # plan with embedded JSON
            "Final Answer: alpha done",
            "Final Answer: beta done",
            "Final Answer: complete",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm)
        result = agent.run("test embedded json")
        assert result.metadata["plan"] == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# RouterAgent tests
# ---------------------------------------------------------------------------


class TestRouterAgent:
    def _make_sub_agents(self):
        math_llm = MockLLMProvider(responses=["Final Answer: 4"] * 5)
        chat_llm = MockLLMProvider(responses=["Hello!"] * 5)
        return {
            "math": ReActAgent(math_llm, tools=[CalculatorTool()]),
            "chat": ReActAgent(chat_llm),
        }

    def test_routes_to_correct_agent(self):
        sub_agents = self._make_sub_agents()
        router_llm = MockLLMProvider(responses=["Route to: math"])
        router = RouterAgent(
            llm=router_llm,
            agents=sub_agents,
            descriptions={"math": "Math tasks", "chat": "General chat"},
        )
        result = router.run("What is 2 + 2?")
        assert result.output == "4"
        assert result.metadata["routed_to"] == "math"

    def test_routes_to_chat_agent(self):
        sub_agents = self._make_sub_agents()
        router_llm = MockLLMProvider(responses=["Route to: chat"])
        router = RouterAgent(llm=router_llm, agents=sub_agents)
        result = router.run("Say hello")
        assert result.output == "Hello!"
        assert result.metadata["routed_to"] == "chat"

    def test_fallback_used_when_routing_fails(self):
        sub_agents = self._make_sub_agents()
        fallback_llm = MockLLMProvider(responses=["Final Answer: fallback used"])
        fallback = ReActAgent(fallback_llm)
        router_llm = MockLLMProvider(responses=["I don't know"] * 10)
        router = RouterAgent(
            llm=router_llm, agents=sub_agents, fallback=fallback, max_iterations=2
        )
        result = router.run("something ambiguous")
        assert result.metadata["routed_to"] == "fallback"

    def test_no_agents_returns_error(self):
        router = RouterAgent(llm=MockLLMProvider(), agents={})
        result = router.run("anything")
        assert result.metadata.get("error") is not None

    def test_routing_failure_without_fallback(self):
        sub_agents = self._make_sub_agents()
        router_llm = MockLLMProvider(responses=["I cannot decide"] * 10)
        router = RouterAgent(
            llm=router_llm, agents=sub_agents, fallback=None, max_iterations=2
        )
        result = router.run("ambiguous task")
        assert result.metadata.get("routing_failed") is True

    def test_register_adds_agent(self):
        router_llm = MockLLMProvider(responses=["Route to: new"])
        router = RouterAgent(llm=router_llm, agents={})
        new_agent = ReActAgent(MockLLMProvider(responses=["Final Answer: new agent"]))
        router.register("new", new_agent, description="A new agent")
        result = router.run("do something")
        assert result.metadata["routed_to"] == "new"

    def test_case_insensitive_routing(self):
        sub_agents = self._make_sub_agents()
        router_llm = MockLLMProvider(responses=["Route to: MATH"])
        router = RouterAgent(llm=router_llm, agents=sub_agents)
        result = router.run("calculate something")
        assert result.metadata["routed_to"] == "math"
