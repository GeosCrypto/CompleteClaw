"""Pipeline workflow – parallel step execution with dependency resolution."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional

from completeclaw.workflows.base import Workflow, WorkflowResult, WorkflowStep


class PipelineStep(WorkflowStep):
    """A workflow step that can declare dependencies on other steps.

    Parameters
    ----------
    name:
        Unique step name.
    fn:
        ``fn(context) -> Any``
    depends_on:
        Names of steps that must complete before this one starts.
    description:
        Optional human-readable description.
    """

    def __init__(
        self,
        name: str,
        fn: Any,
        *,
        depends_on: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        super().__init__(name=name, fn=fn, description=description)
        self.depends_on: List[str] = depends_on or []


class Pipeline(Workflow):
    """A pipeline that runs independent steps concurrently.

    Steps that have no dependencies (or whose dependencies have already
    completed) are submitted to a thread-pool executor and run in parallel.
    This is particularly useful for I/O-bound tasks such as simultaneous
    LLM calls.

    Parameters
    ----------
    steps:
        List of :class:`PipelineStep` objects.
    max_workers:
        Thread-pool size.  Defaults to the number of steps.
    name:
        Optional pipeline identifier.

    Example::

        from completeclaw.workflows.pipeline import Pipeline, PipelineStep

        steps = [
            PipelineStep("a", lambda ctx: "result_a"),
            PipelineStep("b", lambda ctx: "result_b"),
            PipelineStep("c", lambda ctx: ctx["a"] + "+" + ctx["b"],
                         depends_on=["a", "b"]),
        ]
        pipeline = Pipeline(steps)
        result = pipeline.run()
        print(result.get("c"))   # "result_a+result_b"
    """

    name = "pipeline"

    def __init__(
        self,
        steps: Optional[List[PipelineStep]] = None,
        *,
        max_workers: Optional[int] = None,
        name: str = "",
    ) -> None:
        self._steps: List[PipelineStep] = list(steps or [])
        self._max_workers = max_workers
        if name:
            self.name = name

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Append a step (returns self for chaining)."""
        self._steps.append(step)
        return self

    def run(
        self, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> WorkflowResult:
        ctx: Dict[str, Any] = dict(context or {})
        ctx.update(kwargs)

        step_map = {s.name: s for s in self._steps}
        completed: Dict[str, Any] = {}
        pending = list(self._steps)

        max_workers = self._max_workers or max(1, len(self._steps))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[concurrent.futures.Future, str] = {}

            def _submit_ready() -> None:
                for step in list(pending):
                    if all(dep in completed for dep in step.depends_on):
                        pending.remove(step)
                        run_ctx = {**ctx, **completed}
                        fut = executor.submit(step.fn, run_ctx)
                        futures[fut] = step.name

            _submit_ready()

            while futures:
                done, _ = concurrent.futures.wait(
                    futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    step_name = futures.pop(fut)
                    result = fut.result()  # re-raises on exception
                    completed[step_name] = result
                    ctx[step_name] = result
                _submit_ready()

        return WorkflowResult(
            outputs=ctx,
            metadata={"workflow": self.name, "steps_run": list(completed.keys())},
        )

    def __repr__(self) -> str:
        steps = [s.name for s in self._steps]
        return f"Pipeline(steps={steps!r})"
