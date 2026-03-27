"""Tests for workflow implementations."""

from __future__ import annotations

import pytest

from completeclaw.workflows.base import WorkflowResult, WorkflowStep
from completeclaw.workflows.chain import SequentialChain
from completeclaw.workflows.pipeline import Pipeline, PipelineStep

# ---------------------------------------------------------------------------
# SequentialChain
# ---------------------------------------------------------------------------


class TestSequentialChain:
    def test_empty_chain(self):
        chain = SequentialChain()
        result = chain.run({"x": 1})
        assert isinstance(result, WorkflowResult)
        assert result.get("x") == 1

    def test_basic_chain(self):
        steps = [
            WorkflowStep("double", lambda ctx: ctx["value"] * 2),
            WorkflowStep("add_one", lambda ctx: ctx["double"] + 1),
        ]
        chain = SequentialChain(steps)
        result = chain.run({"value": 5})
        assert result.get("double") == 10
        assert result.get("add_one") == 11

    def test_step_has_access_to_previous(self):
        steps = [
            WorkflowStep("greet", lambda ctx: "Hello, " + ctx.get("name", "world")),
            WorkflowStep("shout", lambda ctx: ctx["greet"].upper()),
        ]
        chain = SequentialChain(steps)
        result = chain.run({"name": "Alice"})
        assert result.get("shout") == "HELLO, ALICE"

    def test_missing_context_key_raises(self):
        steps = [WorkflowStep("oops", lambda ctx: ctx["does_not_exist"])]
        chain = SequentialChain(steps)
        with pytest.raises(RuntimeError, match="oops"):
            chain.run({})

    def test_add_step_fluent(self):
        chain = (
            SequentialChain()
            .add_step(WorkflowStep("a", lambda ctx: 1))
            .add_step(WorkflowStep("b", lambda ctx: ctx["a"] + 1))
        )
        result = chain.run()
        assert result.get("b") == 2

    def test_context_passthrough_via_kwargs(self):
        steps = [WorkflowStep("echo", lambda ctx: ctx["msg"])]
        chain = SequentialChain(steps)
        result = chain.run(msg="passed via kwargs")
        assert result.get("echo") == "passed via kwargs"

    def test_metadata_contains_step_names(self):
        steps = [WorkflowStep("s1", lambda ctx: "ok")]
        chain = SequentialChain(steps)
        result = chain.run()
        step_names = [s["name"] for s in result.metadata.get("steps", [])]
        assert "s1" in step_names

    def test_custom_name(self):
        chain = SequentialChain(name="my_chain")
        assert chain.name == "my_chain"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_parallel_independent_steps(self):
        steps = [
            PipelineStep("a", lambda ctx: "A"),
            PipelineStep("b", lambda ctx: "B"),
            PipelineStep("c", lambda ctx: "C"),
        ]
        pipeline = Pipeline(steps)
        result = pipeline.run()
        assert result.get("a") == "A"
        assert result.get("b") == "B"
        assert result.get("c") == "C"

    def test_dependency_ordering(self):
        steps = [
            PipelineStep("x", lambda ctx: 10),
            PipelineStep("y", lambda ctx: 20),
            PipelineStep("z", lambda ctx: ctx["x"] + ctx["y"], depends_on=["x", "y"]),
        ]
        pipeline = Pipeline(steps)
        result = pipeline.run()
        assert result.get("z") == 30

    def test_initial_context(self):
        steps = [PipelineStep("out", lambda ctx: ctx["val"] * 3)]
        pipeline = Pipeline(steps)
        result = pipeline.run({"val": 7})
        assert result.get("out") == 21

    def test_exception_propagates(self):
        steps = [PipelineStep("boom", lambda ctx: 1 / 0)]
        pipeline = Pipeline(steps)
        with pytest.raises(ZeroDivisionError):
            pipeline.run()

    def test_add_step_fluent(self):
        pipeline = (
            Pipeline()
            .add_step(PipelineStep("a", lambda ctx: "hello"))
        )
        result = pipeline.run()
        assert result.get("a") == "hello"

    def test_metadata(self):
        steps = [PipelineStep("s", lambda ctx: "v")]
        pipeline = Pipeline(steps, name="my_pipeline")
        result = pipeline.run()
        assert result.metadata["workflow"] == "my_pipeline"
        assert "s" in result.metadata["steps_run"]


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_get_default(self):
        r = WorkflowResult()
        assert r.get("missing") is None
        assert r.get("missing", "default") == "default"

    def test_repr(self):
        r = WorkflowResult(outputs={"a": 1, "b": 2})
        assert "WorkflowResult" in repr(r)
