"""Parallel pipeline workflow with dependency resolution."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Set

from completeclaw.workflows.base import Workflow, WorkflowResult, WorkflowStep


class _PipelineStep(WorkflowStep):
    """Internal step with dependency tracking."""

    def __init__(
        self,
        name: str,
        fn: Any,
        description: str = "",
        depends_on: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name, fn=fn, description=description)
        self.depends_on: List[str] = depends_on or []


class Pipeline(Workflow):
    """Parallel pipeline that resolves step dependencies automatically.

    Steps without dependencies (or whose dependencies are satisfied) are
    executed concurrently using threads.

    Parameters
    ----------
    name:
        Name of this pipeline.

    Example::

        p = Pipeline()
        p.add_step("fetch", fn=lambda ctx: "raw data")
        p.add_step("parse", fn=lambda ctx: ctx["fetch"].upper(), depends_on=["fetch"])
        result = p.run()
        print(result.get("parse"))  # "RAW DATA"
    """

    def __init__(self, *, name: str = "pipeline") -> None:
        self.name = name
        self._steps: Dict[str, _PipelineStep] = {}

    def add_step(
        self,
        name: str,
        fn: Any,
        *,
        description: str = "",
        depends_on: Optional[List[str]] = None,
    ) -> "Pipeline":
        """Add a step to the pipeline and return *self* for chaining."""
        self._steps[name] = _PipelineStep(
            name=name, fn=fn, description=description, depends_on=depends_on or []
        )
        return self

    def run(self, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WorkflowResult:
        ctx: Dict[str, Any] = dict(context or {})
        completed: Set[str] = set()
        lock = threading.Lock()

        remaining = set(self._steps)

        while remaining:
            # Find steps whose dependencies are all completed
            ready = [
                name
                for name in remaining
                if all(dep in completed for dep in self._steps[name].depends_on)
            ]
            if not ready:
                raise RuntimeError(
                    f"Pipeline deadlock – unresolvable dependencies for: {remaining}"
                )

            # Run ready steps in parallel
            threads = []
            results: Dict[str, Any] = {}

            def _run(step_name: str) -> None:
                with lock:
                    local_ctx = dict(ctx)
                output = self._steps[step_name].fn(local_ctx)
                with lock:
                    results[step_name] = output

            for name in ready:
                t = threading.Thread(target=_run, args=(name,), daemon=True)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            with lock:
                ctx.update(results)
                completed.update(ready)
                remaining -= set(ready)

        return WorkflowResult(outputs=ctx)
