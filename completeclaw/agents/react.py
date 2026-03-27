"""ReAct (Reason + Act) agent implementation.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.11610
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from completeclaw.agents.base import Agent, AgentResult
from completeclaw.llm.base import Message, Role
from completeclaw.memory.base import Memory
from completeclaw.tools.base import Tool


_SYSTEM_TEMPLATE = """\
You are a helpful AI assistant that reasons step-by-step and uses tools when needed.

Available tools:
{tool_descriptions}

To use a tool respond with:
Action: <tool_name>
Action Input: <JSON object with the tool's input>

When you have a final answer respond with:
Final Answer: <your answer>

Always think before acting:
Thought: <your reasoning>
"""

_NO_TOOLS_SYSTEM = """\
You are a helpful AI assistant that reasons step-by-step.

When you have a final answer respond with:
Final Answer: <your answer>
"""


class ReActAgent(Agent):
    """A ReAct-style agent that interleaves reasoning and tool use.

    The agent iterates through Thought → Action → Observation cycles
    until it reaches a ``Final Answer`` or the ``max_iterations`` limit.

    Example::

        from completeclaw.llm.mock import MockLLMProvider
        from completeclaw.tools.calculator import CalculatorTool
        from completeclaw.agents.react import ReActAgent

        llm = MockLLMProvider(responses=["Final Answer: 42"])
        agent = ReActAgent(llm, tools=[CalculatorTool()])
        result = agent.run("What is 6 * 7?")
        print(result.output)   # "42"
    """

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        steps: List[Dict[str, Any]] = []
        messages: List[Message] = []

        # Build system prompt
        if self.tool_registry.tools:
            tool_desc = "\n".join(
                f"- {name}: {tool.description}"
                for name, tool in self.tool_registry.tools.items()
            )
            system_content = _SYSTEM_TEMPLATE.format(tool_descriptions=tool_desc)
        else:
            system_content = _NO_TOOLS_SYSTEM

        if self.system_prompt:
            system_content = self.system_prompt + "\n\n" + system_content

        messages.append(Message(role=Role.SYSTEM, content=system_content))

        # Inject memory context if available
        if self.memory is not None:
            history = self.memory.load()
            for entry in history:
                messages.append(
                    Message(role=Role.USER if entry.role == "user" else Role.ASSISTANT,
                            content=entry.content)
                )

        messages.append(Message(role=Role.USER, content=task))

        for iteration in range(self.max_iterations):
            response = self.llm.chat(messages, **kwargs)
            reply = response.content.strip()
            messages.append(Message(role=Role.ASSISTANT, content=reply))

            # Check for final answer
            final_match = re.search(
                r"Final Answer\s*:\s*(.+)", reply, re.IGNORECASE | re.DOTALL
            )
            if final_match:
                output = final_match.group(1).strip()
                if self.memory is not None:
                    self.memory.save_user(task)
                    self.memory.save_assistant(output)
                return AgentResult(
                    output=output,
                    steps=steps,
                    metadata={"iterations": iteration + 1},
                )

            # Parse Action / Action Input
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
                    tool_input = json.loads(tool_input_str)
                except json.JSONDecodeError:
                    tool_input = {"input": tool_input_str}

                step: Dict[str, Any] = {
                    "iteration": iteration + 1,
                    "thought": _extract_thought(reply),
                    "action": tool_name,
                    "action_input": tool_input,
                }

                tool = self.tool_registry.get(tool_name)
                if tool is None:
                    observation = f"Error: tool '{tool_name}' not found."
                else:
                    result = tool.run(**tool_input)
                    observation = result.output if result.success else f"Error: {result.error}"

                step["observation"] = observation
                steps.append(step)

                messages.append(
                    Message(role=Role.USER, content=f"Observation: {observation}")
                )
            else:
                # No action found – treat the whole reply as the final answer
                if self.memory is not None:
                    self.memory.save_user(task)
                    self.memory.save_assistant(reply)
                return AgentResult(
                    output=reply,
                    steps=steps,
                    metadata={"iterations": iteration + 1},
                )

        return AgentResult(
            output="Maximum iterations reached without a final answer.",
            steps=steps,
            metadata={"iterations": self.max_iterations, "truncated": True},
        )


def _extract_thought(text: str) -> str:
    """Extract the Thought portion of a ReAct reply."""
    match = re.search(r"Thought\s*:\s*(.+?)(?=\nAction|\nFinal Answer|\Z)",
                      text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""
