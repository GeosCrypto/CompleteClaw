"""Tests for workflow orchestration."""

from __future__ import annotations

import pytest

from completeclaw.workflows.base import WorkflowResult, WorkflowStep
from completeclaw.workflows.chain import SequentialChain
from completeclaw.workflows.pipeline import Pipeline


class TestWorkflowResult:
    def test_get_returns_value(self):
        r = WorkflowResult(outputs={"key": "value"})
        assert r.get("key") == "value"

    def test_get_returns_default(self):
        r = WorkflowResult()
        assert r.get("missing", "default") == "default"

    def test_repr(self):
        r = WorkflowResult(outputs={"a": 1})
        assert "a" in repr(r)


class TestSequentialChain:
    def test_runs_steps_in_order(self):
        chain = SequentialChain(steps=[
            WorkflowStep("upper", fn=lambda ctx: ctx["input"].upper()),
            WorkflowStep("greet", fn=lambda ctx: f"Hello, {ctx['upper']}!"),
        ])
        result = chain.run(context={"input": "world"})
        assert result.get("greet") == "Hello, WORLD!"

    def test_empty_context(self):
        chain = SequentialChain(steps=[
            WorkflowStep("const", fn=lambda ctx: 42),
        ])
        result = chain.run()
        assert result.get("const") == 42

    def test_repr(self):
        chain = SequentialChain(steps=[], name="my_chain")
        assert "SequentialChain" in repr(chain)


class TestPipeline:
    def test_simple_pipeline(self):
        p = Pipeline()
        p.add_step("a", fn=lambda ctx: 1)
        p.add_step("b", fn=lambda ctx: 2)
        result = p.run()
        assert result.get("a") == 1
        assert result.get("b") == 2

    def test_dependency_resolution(self):
        p = Pipeline()
        p.add_step("fetch", fn=lambda ctx: "raw")
        p.add_step("parse", fn=lambda ctx: ctx["fetch"].upper(), depends_on=["fetch"])
        result = p.run()
        assert result.get("parse") == "RAW"

    def test_chained_dependencies(self):
        p = Pipeline()
        p.add_step("a", fn=lambda ctx: 1)
        p.add_step("b", fn=lambda ctx: ctx["a"] + 1, depends_on=["a"])
        p.add_step("c", fn=lambda ctx: ctx["b"] + 1, depends_on=["b"])
        result = p.run()
        assert result.get("c") == 3

    def test_cyclic_dependency_raises(self):
        p = Pipeline()
        p.add_step("a", fn=lambda ctx: 1, depends_on=["b"])
        p.add_step("b", fn=lambda ctx: 2, depends_on=["a"])
        with pytest.raises(RuntimeError, match="deadlock"):
            p.run()

    def test_repr(self):
        p = Pipeline(name="my_pipeline")
        assert "Pipeline" in repr(p)
