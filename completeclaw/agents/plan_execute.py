"""Plan-and-Execute agent implementation.

This agent first generates a step-by-step plan for a task and then
executes each step using a ReAct-style reasoning loop, making it
well-suited for complex, multi-step tasks that span diverse domains.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from completeclaw.agents.base import Agent, AgentResult
from completeclaw.llm.base import Message, Role

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a planning assistant. Given a task, break it down into a clear,
ordered list of concrete steps needed to complete it. Each step should be
self-contained and actionable.

Respond with ONLY a JSON array of step strings, for example:
["Step 1 description", "Step 2 description", "Step 3 description"]
"""

_EXECUTE_SYSTEM_TEMPLATE = """\
You are a helpful AI assistant that reasons step-by-step and uses tools \
when needed.

Available tools:
{tool_descriptions}

To use a tool respond with:
Action: <tool_name>
Action Input: <JSON object with the tool's input>

When you have a final answer for the current step respond with:
Final Answer: <your answer>

Always think before acting:
Thought: <your reasoning>

Overall task context:
{task_context}
"""

_EXECUTE_SYSTEM_NO_TOOLS = """\
You are a helpful AI assistant that reasons step-by-step.

When you have a final answer for the current step respond with:
Final Answer: <your answer>

Overall task context:
{task_context}
"""

_SYNTHESIS_SYSTEM = """\
You are a synthesis assistant. Given the original task and the results of
each completed step, produce a single, coherent final answer.

Respond with:
Final Answer: <your consolidated answer>
"""


class PlanAndExecuteAgent(Agent):
    """An agent that plans a task before executing it step by step.

    The agent operates in three phases:

    1. **Planning** – The LLM decomposes the task into an ordered list of
       concrete steps.
    2. **Execution** – Each step is executed with a ReAct-style
       Thought → Action → Observation loop.  Results are collected.
    3. **Synthesis** – Step results are consolidated into a single final
       answer.

    This design is well suited for complex, multi-domain tasks where
    upfront planning leads to more reliable and thorough execution.

    Parameters
    ----------
    llm:
        The LLM provider used for planning, execution, and synthesis.
    tools:
        Optional list of tools available during the execution phase.
    memory:
        Optional memory store for persisting context across sessions.
    max_iterations:
        Maximum Thought/Action/Observation cycles per step.
    system_prompt:
        Optional extra instructions prepended to every system message.
    max_plan_steps:
        Upper bound on the number of plan steps (excess steps are dropped).

    Example::

        from completeclaw.llm.mock import MockLLMProvider
        from completeclaw.agents.plan_execute import PlanAndExecuteAgent

        responses = [
            '["Gather data", "Analyse data", "Report findings"]',
            "Final Answer: data gathered",
            "Final Answer: analysis done",
            "Final Answer: All steps complete",
        ]
        llm = MockLLMProvider(responses=responses)
        agent = PlanAndExecuteAgent(llm, max_plan_steps=5)
        result = agent.run("Research and summarise quantum computing trends")
        print(result.output)
    """

    def __init__(
        self,
        llm: Any,
        *,
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
        max_plan_steps: int = 10,
    ) -> None:
        super().__init__(
            llm,
            tools=tools,
            memory=memory,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )
        self.max_plan_steps = max_plan_steps

    # ------------------------------------------------------------------
    # Planning phase
    # ------------------------------------------------------------------

    def _make_plan(self, task: str) -> List[str]:
        """Ask the LLM to break *task* into an ordered list of steps.

        Returns a list of step strings.  Falls back to a single-step plan
        containing the original task if parsing fails.
        """
        system_content = _PLAN_SYSTEM
        if self.system_prompt:
            system_content = self.system_prompt + "\n\n" + system_content

        messages = [
            Message(role=Role.SYSTEM, content=system_content),
            Message(role=Role.USER, content=f"Task: {task}"),
        ]
        response = self.llm.chat(messages)
        raw = response.content.strip()

        # Try to parse the full response as a JSON array
        try:
            steps = json.loads(raw)
            if isinstance(steps, list) and steps:
                return [str(s) for s in steps[: self.max_plan_steps]]
        except json.JSONDecodeError:
            pass

        # Fallback: look for a JSON array embedded in the text
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                steps = json.loads(match.group())
                if isinstance(steps, list) and steps:
                    return [str(s) for s in steps[: self.max_plan_steps]]
            except json.JSONDecodeError:
                pass

        # Last resort: treat the whole reply as one step
        return [task]

    # ------------------------------------------------------------------
    # Execution phase
    # ------------------------------------------------------------------

    def _execute_step(self, step: str, task_context: str) -> str:
        """Execute a single plan *step* using a ReAct-style loop.

        Parameters
        ----------
        step:
            The step description to execute.
        task_context:
            A brief reminder of the overall task, injected into the system
            prompt so the LLM keeps the bigger picture in mind.

        Returns
        -------
        str
            The observation or final answer produced for this step.
        """
        if self.tool_registry.tools:
            tool_desc = "\n".join(
                f"- {name}: {tool.description}"
                for name, tool in self.tool_registry.tools.items()
            )
            system_content = _EXECUTE_SYSTEM_TEMPLATE.format(
                tool_descriptions=tool_desc,
                task_context=task_context,
            )
        else:
            system_content = _EXECUTE_SYSTEM_NO_TOOLS.format(
                task_context=task_context,
            )

        messages: List[Message] = [
            Message(role=Role.SYSTEM, content=system_content),
            Message(role=Role.USER, content=step),
        ]

        for _ in range(self.max_iterations):
            response = self.llm.chat(messages)
            reply = response.content.strip()
            messages.append(Message(role=Role.ASSISTANT, content=reply))

            # Final answer for this step?
            final_match = re.search(
                r"Final Answer\s*:\s*(.+)", reply, re.IGNORECASE | re.DOTALL
            )
            if final_match:
                return final_match.group(1).strip()

            # Tool call?
            action_match = re.search(r"Action\s*:\s*(\S+)", reply, re.IGNORECASE)
            input_match = re.search(
                r"Action Input\s*:\s*(.+?)(?=\nThought|\nAction|\Z)",
                reply,
                re.IGNORECASE | re.DOTALL,
            )

            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input_str = input_match.group(1).strip() if input_match else "{}"
                try:
                    tool_input: Dict[str, Any] = json.loads(tool_input_str)
                except json.JSONDecodeError:
                    tool_input = {"input": tool_input_str}

                tool = self.tool_registry.get(tool_name)
                if tool is None:
                    observation = f"Error: tool '{tool_name}' not found."
                else:
                    result = tool.run(**tool_input)
                    observation = result.output if result.success else f"Error: {result.error}"

                messages.append(
                    Message(role=Role.USER, content=f"Observation: {observation}")
                )
            else:
                # No action or final answer – return reply as-is
                return reply

        return "Maximum iterations reached for this step."

    # ------------------------------------------------------------------
    # Synthesis phase
    # ------------------------------------------------------------------

    def _synthesise(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        """Combine per-step results into one coherent final answer."""
        summary_lines = [f"Task: {task}", "", "Step results:"]
        for i, sr in enumerate(step_results, 1):
            summary_lines.append(f"  Step {i} ({sr['step']}): {sr['result']}")

        synthesis_prompt = "\n".join(summary_lines)

        system_content = _SYNTHESIS_SYSTEM
        if self.system_prompt:
            system_content = self.system_prompt + "\n\n" + system_content

        messages = [
            Message(role=Role.SYSTEM, content=system_content),
            Message(role=Role.USER, content=synthesis_prompt),
        ]
        response = self.llm.chat(messages)
        raw = response.content.strip()

        final_match = re.search(
            r"Final Answer\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL
        )
        return final_match.group(1).strip() if final_match else raw

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute *task* via plan-then-execute.

        Parameters
        ----------
        task:
            The high-level goal for the agent to accomplish.

        Returns
        -------
        AgentResult
            Contains the synthesised final answer, per-step execution
            trace, and metadata (``plan``, ``plan_steps``).
        """
        steps_trace: List[Dict[str, Any]] = []

        # Inject memory history as context if available
        if self.memory is not None:
            history = self.memory.load()
            history_text = " | ".join(e.content for e in history[-4:]) if history else ""
            task_context = f"{history_text}\n{task}".strip() if history_text else task
        else:
            task_context = task

        # Phase 1: plan
        plan = self._make_plan(task)

        # Phase 2: execute each step
        step_results: List[Dict[str, Any]] = []
        for i, step in enumerate(plan):
            result_text = self._execute_step(step, task_context)
            step_results.append({"step": step, "result": result_text})
            steps_trace.append(
                {
                    "plan_step": i + 1,
                    "step": step,
                    "result": result_text,
                }
            )

        # Phase 3: synthesise
        final_output = self._synthesise(task, step_results)

        # Persist to memory if configured
        if self.memory is not None:
            self.memory.save_user(task)
            self.memory.save_assistant(final_output)

        return AgentResult(
            output=final_output,
            steps=steps_trace,
            metadata={
                "plan": plan,
                "plan_steps": len(plan),
            },
        )
