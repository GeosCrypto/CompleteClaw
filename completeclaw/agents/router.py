"""Router agent that delegates tasks to specialised sub-agents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from completeclaw.agents.base import Agent, AgentResult
from completeclaw.llm.base import LLMProvider, Message, Role


class RouterAgent(Agent):
    """An agent that delegates tasks to specialised sub-agents.

    The ``RouterAgent`` maintains a registry of named sub-agents and uses
    its LLM to decide which agent is best suited for each incoming task.
    The LLM is instructed to respond with ``"Route to: <agent_name>"``;
    the named agent is then called with the original task.

    Parameters
    ----------
    llm:
        LLM provider used for routing decisions.
    agents:
        ``dict`` mapping name → :class:`~completeclaw.agents.base.Agent`, or a
        list of ``(name, Agent)`` pairs.
    descriptions:
        Optional ``dict`` mapping name → human-readable description shown to
        the LLM.  If omitted, the agent's class name is used.
    fallback:
        Optional fallback :class:`~completeclaw.agents.base.Agent` used when
        routing fails.  If *None*, an error :class:`AgentResult` is returned.
    system_prompt:
        Additional instruction prepended to the routing system prompt.
    max_iterations:
        How many times the router LLM may be re-prompted before giving up
        (distinct from the sub-agent's own ``max_iterations``).

    Example::

        from completeclaw.llm.mock import MockLLMProvider
        from completeclaw.agents.react import ReActAgent
        from completeclaw.agents.router import RouterAgent
        from completeclaw.tools.calculator import CalculatorTool

        math_agent = ReActAgent(
            MockLLMProvider(responses=["Final Answer: 4"]),
            tools=[CalculatorTool()],
        )
        chat_agent = ReActAgent(MockLLMProvider(responses=["Hello!"]))

        router = RouterAgent(
            llm=MockLLMProvider(responses=["Route to: math"]),
            agents={"math": math_agent, "chat": chat_agent},
            descriptions={"math": "Handles calculations", "chat": "General conversation"},
        )
        result = router.run("What is 2 + 2?")
        print(result.output)            # "4"
        print(result.metadata["routed_to"])  # "math"
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        agents: Union[Dict[str, Agent], List[Tuple[str, Agent]]],
        descriptions: Optional[Dict[str, str]] = None,
        fallback: Optional[Agent] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 3,
    ) -> None:
        self._agents: Dict[str, Agent] = dict(agents)  # type: ignore[arg-type]
        self._descriptions: Dict[str, str] = dict(descriptions or {})
        self._fallback = fallback
        # RouterAgent itself has no tools or memory of its own
        super().__init__(
            llm,
            tools=None,
            memory=None,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register(self, name: str, agent: Agent, description: str = "") -> None:
        """Register a new sub-agent at runtime."""
        self._agents[name] = agent
        if description:
            self._descriptions[name] = description

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        if not self._agents:
            return AgentResult(
                output="No sub-agents registered.",
                metadata={"routed_to": None, "error": "No agents available"},
            )

        chosen = self._route(task, **kwargs)
        if chosen is None:
            if self._fallback is not None:
                result = self._fallback.run(task, **kwargs)
                result.metadata["routed_to"] = "fallback"
                return result
            return AgentResult(
                output="Could not determine which agent should handle this task.",
                metadata={"routed_to": None, "routing_failed": True},
            )

        result = self._agents[chosen].run(task, **kwargs)
        result.metadata["routed_to"] = chosen
        return result

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route(self, task: str, **kwargs: Any) -> Optional[str]:
        """Prompt the LLM to choose a sub-agent; return None on failure."""
        agent_list = "\n".join(
            f"- {name}: {self._descriptions.get(name, type(agent).__name__)}"
            for name, agent in self._agents.items()
        )
        system = (
            "You are a routing assistant. "
            "Given a task and a list of agents, pick the most appropriate agent.\n"
            "Respond with ONLY: Route to: <agent_name>\n\n"
            f"Available agents:\n{agent_list}"
        )
        if self.system_prompt:
            system = self.system_prompt + "\n\n" + system

        messages: List[Message] = [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=task),
        ]

        for _ in range(self.max_iterations):
            response = self.llm.chat(messages, **kwargs)
            reply = response.content.strip()

            match = re.search(r"Route\s+to\s*:\s*(\S+)", reply, re.IGNORECASE)
            if match:
                name = match.group(1).strip().rstrip(".,;")
                if name in self._agents:
                    return name
                # Case-insensitive fallback
                lower = name.lower()
                for key in self._agents:
                    if key.lower() == lower:
                        return key

            messages.append(Message(role=Role.ASSISTANT, content=reply))
            messages.append(
                Message(
                    role=Role.USER,
                    content=(
                        "Please specify which agent to use. "
                        f"Valid names are: {', '.join(self._agents)}. "
                        "Respond with: Route to: <agent_name>"
                    ),
                )
            )

        return None
