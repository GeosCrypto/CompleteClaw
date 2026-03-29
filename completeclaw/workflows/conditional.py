"""Conditional branching workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from completeclaw.workflows.base import Workflow, WorkflowResult


@dataclass
class ConditionalStep:
    """A workflow step that is optionally guarded by a condition.

    Parameters
    ----------
    name:
        Unique key under which the step's output is stored in the context.
    fn:
        Callable ``(ctx: dict) -> Any`` that computes the step's output.
    condition:
        Optional callable ``(ctx: dict) -> bool``.  When provided, the step
        is executed only if the condition is truthy.  Skipped steps leave no
        entry in the context dict.
    description:
        Human-readable description of this step.
    """

    name: str
    fn: Callable[[Dict[str, Any]], Any]
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""


@dataclass
class BranchStep:
    """An if/else branch point in a :class:`ConditionalChain`.

    Evaluates ``condition(ctx)`` and runs one of two sub-sequences.
    The boolean result is stored in the context under ``name``.

    Parameters
    ----------
    name:
        Key under which the branch outcome (``True`` / ``False``) is stored.
    condition:
        Callable ``(ctx: dict) -> bool`` that decides the branch.
    if_steps:
        Sub-steps to execute when the condition is *True*.
    else_steps:
        Sub-steps to execute when the condition is *False*.
    description:
        Human-readable description of this branch.
    """

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    if_steps: List[Union[ConditionalStep, "BranchStep"]] = field(default_factory=list)
    else_steps: List[Union[ConditionalStep, "BranchStep"]] = field(default_factory=list)
    description: str = ""


class ConditionalChain(Workflow):
    """Sequential workflow with conditional step execution and if/else branching.

    Steps are executed in order.  A :class:`ConditionalStep` whose
    ``condition`` callable returns ``False`` is silently skipped.  A
    :class:`BranchStep` evaluates its condition and executes either its
    ``if_steps`` or ``else_steps`` sub-sequence.  Branches may be nested
    arbitrarily.

    Parameters
    ----------
    steps:
        Ordered list of :class:`ConditionalStep` or :class:`BranchStep` objects.
    name:
        Optional name for this chain.

    Example::

        from completeclaw.workflows.conditional import (
            BranchStep,
            ConditionalChain,
            ConditionalStep,
        )

        chain = ConditionalChain(
            steps=[
                ConditionalStep("score", fn=lambda ctx: int(ctx["raw"])),
                BranchStep(
                    name="grade",
                    condition=lambda ctx: ctx["score"] >= 60,
                    if_steps=[ConditionalStep("result", fn=lambda ctx: "Pass")],
                    else_steps=[ConditionalStep("result", fn=lambda ctx: "Fail")],
                ),
                # Only add a note when the result is "Pass"
                ConditionalStep(
                    "note",
                    fn=lambda ctx: "Well done!",
                    condition=lambda ctx: ctx.get("result") == "Pass",
                ),
            ]
        )

        r = chain.run(context={"raw": "75"})
        print(r.get("result"))  # "Pass"
        print(r.get("grade"))   # True
        print(r.get("note"))    # "Well done!"
    """

    def __init__(
        self,
        steps: List[Union[ConditionalStep, BranchStep]],
        *,
        name: str = "conditional_chain",
    ) -> None:
        self.steps = steps
        self.name = name

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        ctx: Dict[str, Any] = dict(context or {})
        self._run_steps(self.steps, ctx)
        return WorkflowResult(outputs=ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_steps(
        self, steps: List[Union[ConditionalStep, BranchStep]], ctx: Dict[str, Any]
    ) -> None:
        for step in steps:
            if isinstance(step, BranchStep):
                result = step.condition(ctx)
                ctx[step.name] = result
                branch = step.if_steps if result else step.else_steps
                self._run_steps(branch, ctx)
            elif isinstance(step, ConditionalStep):
                if step.condition is None or step.condition(ctx):
                    ctx[step.name] = step.fn(ctx)

    def __repr__(self) -> str:
        return f"ConditionalChain(name={self.name!r}, steps={len(self.steps)})"
