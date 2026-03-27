"""Abstract base classes for workflows."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowResult:
    """Aggregated result of running a workflow."""

    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.outputs.get(key, default)

    def __repr__(self) -> str:
        return f"WorkflowResult(outputs={list(self.outputs.keys())!r})"


@dataclass
class WorkflowStep:
    """A single named step within a workflow.

    Parameters
    ----------
    name:
        Unique identifier for this step.
    fn:
        Callable that accepts a ``context`` dict and returns a value.
    description:
        Optional human-readable description.
    """

    name: str
    fn: Any  # Callable[[Dict[str, Any]], Any]
    description: str = ""


class Workflow(abc.ABC):
    """Abstract base class for all workflows.

    A workflow composes multiple steps (tools, agents, LLM calls, …) into
    a reusable, configurable execution plan.

    Example::

        class MyWorkflow(Workflow):
            def run(self, context, **kwargs):
                ...
                return WorkflowResult(outputs={"result": "done"})
    """

    name: str = "workflow"

    @abc.abstractmethod
    def run(self, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> WorkflowResult:
        """Execute the workflow and return a :class:`WorkflowResult`."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
