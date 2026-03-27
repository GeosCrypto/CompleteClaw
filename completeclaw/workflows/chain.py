"""Sequential chain workflow – runs steps one after another."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from completeclaw.workflows.base import Workflow, WorkflowResult, WorkflowStep


class SequentialChain(Workflow):
    """Executes :class:`~completeclaw.workflows.base.WorkflowStep` objects in order.

    Each step receives a shared *context* dict.  The return value of each
    step is stored in ``context[step.name]`` for downstream steps to use.

    Parameters
    ----------
    steps:
        Ordered list of :class:`WorkflowStep` objects.
    name:
        Optional workflow identifier.

    Example::

        from completeclaw.workflows.chain import SequentialChain
        from completeclaw.workflows.base import WorkflowStep

        steps = [
            WorkflowStep("greet", lambda ctx: "Hello, " + ctx.get("name", "world")),
            WorkflowStep("shout", lambda ctx: ctx["greet"].upper()),
        ]
        chain = SequentialChain(steps)
        result = chain.run({"name": "Alice"})
        print(result.get("shout"))   # "HELLO, ALICE"
    """

    name = "sequential_chain"

    def __init__(self, steps: Optional[List[WorkflowStep]] = None, *, name: str = "") -> None:
        self._steps: List[WorkflowStep] = list(steps or [])
        if name:
            self.name = name

    def add_step(self, step: WorkflowStep) -> "SequentialChain":
        """Append a step to the chain (returns self for chaining)."""
        self._steps.append(step)
        return self

    def run(
        self, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> WorkflowResult:
        ctx: Dict[str, Any] = dict(context or {})
        ctx.update(kwargs)

        step_meta: List[Dict[str, Any]] = []

        for step in self._steps:
            try:
                output = step.fn(ctx)
                ctx[step.name] = output
                step_meta.append({"name": step.name, "success": True})
            except Exception as exc:
                step_meta.append({"name": step.name, "success": False, "error": str(exc)})
                raise RuntimeError(
                    f"Step '{step.name}' failed: {exc}"
                ) from exc

        return WorkflowResult(
            outputs=ctx,
            metadata={"steps": step_meta, "workflow": self.name},
        )

    def __repr__(self) -> str:
        steps = [s.name for s in self._steps]
        return f"SequentialChain(steps={steps!r})"
