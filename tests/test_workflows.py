"""Tests for workflow orchestration."""

from __future__ import annotations

import pytest

from completeclaw.workflows.base import WorkflowResult, WorkflowStep
from completeclaw.workflows.chain import SequentialChain
from completeclaw.workflows.conditional import BranchStep, ConditionalChain, ConditionalStep
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


# ---------------------------------------------------------------------------
# ConditionalChain tests
# ---------------------------------------------------------------------------


class TestConditionalChain:
    def test_unconditional_step_always_runs(self):
        chain = ConditionalChain(steps=[
            ConditionalStep("val", fn=lambda ctx: 42),
        ])
        result = chain.run()
        assert result.get("val") == 42

    def test_condition_true_step_runs(self):
        chain = ConditionalChain(steps=[
            ConditionalStep("x", fn=lambda ctx: 10),
            ConditionalStep(
                "doubled",
                fn=lambda ctx: ctx["x"] * 2,
                condition=lambda ctx: ctx["x"] > 5,
            ),
        ])
        result = chain.run()
        assert result.get("doubled") == 20

    def test_condition_false_step_is_skipped(self):
        chain = ConditionalChain(steps=[
            ConditionalStep("x", fn=lambda ctx: 3),
            ConditionalStep(
                "doubled",
                fn=lambda ctx: ctx["x"] * 2,
                condition=lambda ctx: ctx["x"] > 5,
            ),
        ])
        result = chain.run()
        assert result.get("doubled") is None  # skipped

    def test_branch_if_path(self):
        chain = ConditionalChain(steps=[
            BranchStep(
                name="branch",
                condition=lambda ctx: ctx["score"] >= 60,
                if_steps=[ConditionalStep("result", fn=lambda ctx: "Pass")],
                else_steps=[ConditionalStep("result", fn=lambda ctx: "Fail")],
            ),
        ])
        result = chain.run(context={"score": 75})
        assert result.get("result") == "Pass"
        assert result.get("branch") is True

    def test_branch_else_path(self):
        chain = ConditionalChain(steps=[
            BranchStep(
                name="branch",
                condition=lambda ctx: ctx["score"] >= 60,
                if_steps=[ConditionalStep("result", fn=lambda ctx: "Pass")],
                else_steps=[ConditionalStep("result", fn=lambda ctx: "Fail")],
            ),
        ])
        result = chain.run(context={"score": 45})
        assert result.get("result") == "Fail"
        assert result.get("branch") is False

    def test_nested_branches(self):
        chain = ConditionalChain(steps=[
            BranchStep(
                name="outer",
                condition=lambda ctx: ctx["x"] > 0,
                if_steps=[
                    BranchStep(
                        name="inner",
                        condition=lambda ctx: ctx["x"] > 10,
                        if_steps=[ConditionalStep("grade", fn=lambda ctx: "A")],
                        else_steps=[ConditionalStep("grade", fn=lambda ctx: "B")],
                    )
                ],
                else_steps=[ConditionalStep("grade", fn=lambda ctx: "F")],
            )
        ])
        assert chain.run(context={"x": 15}).get("grade") == "A"
        assert chain.run(context={"x": 5}).get("grade") == "B"
        assert chain.run(context={"x": -1}).get("grade") == "F"

    def test_initial_context_is_preserved(self):
        chain = ConditionalChain(steps=[
            ConditionalStep("b", fn=lambda ctx: ctx["a"] + 1),
        ])
        result = chain.run(context={"a": 10})
        assert result.get("a") == 10
        assert result.get("b") == 11

    def test_empty_steps(self):
        chain = ConditionalChain(steps=[])
        result = chain.run(context={"x": 1})
        assert result.get("x") == 1

    def test_repr(self):
        chain = ConditionalChain(steps=[], name="test_chain")
        r = repr(chain)
        assert "ConditionalChain" in r
        assert "test_chain" in r
