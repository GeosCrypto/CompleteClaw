"""Sequential chain workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from completeclaw.workflows.base import Workflow, WorkflowResult, WorkflowStep


class SequentialChain(Workflow):
    """Executes workflow steps one after another, passing context forward.

    Each step receives the shared ``context`` dict (which accumulates
    all previous step outputs) and its return value is stored under the
    step's name.

    Parameters
    ----------
    steps:
        Ordered list of :class:`WorkflowStep` objects.
    name:
        Optional name for this chain.

    Example::

        chain = SequentialChain(steps=[
            WorkflowStep("upper", fn=lambda ctx: ctx["input"].upper()),
            WorkflowStep("greet", fn=lambda ctx: f"Hello, {ctx['upper']}!"),
        ])
        result = chain.run(context={"input": "world"})
        print(result.get("greet"))  # "Hello, WORLD!"
    """

    def __init__(self, steps: List[WorkflowStep], *, name: str = "sequential_chain") -> None:
        self.steps = steps
        self.name = name

    def run(self, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WorkflowResult:
        ctx: Dict[str, Any] = dict(context or {})
        for step in self.steps:
            ctx[step.name] = step.fn(ctx)
        return WorkflowResult(outputs=ctx)
