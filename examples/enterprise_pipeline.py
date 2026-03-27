"""
Example: Enterprise pipeline – multiple LLM steps run in parallel.

This shows how to use the Pipeline workflow to run independent subtasks
concurrently (e.g. summarise different documents simultaneously).

Run:
    python examples/enterprise_pipeline.py
"""

from __future__ import annotations

from completeclaw.llm.mock import MockLLMProvider
from completeclaw.workflows.pipeline import Pipeline, PipelineStep


def make_summariser(llm, doc_key: str):
    """Return a pipeline step that summarises ctx[doc_key]."""

    def step(ctx):
        doc = ctx.get(doc_key, "")
        return llm.complete(f"Summarise in one sentence: {doc}")

    return step


def main() -> None:
    # Simulate 3 LLMs (or use the same one with different prompts)
    llm = MockLLMProvider(
        reply_fn=lambda msgs: f"[Summary of: {msgs[-1].content[:40]}…]"
    )

    documents = {
        "doc_finance": "Q3 revenue grew 12% YoY driven by cloud subscriptions …",
        "doc_legal": "The merger agreement stipulates a 90-day exclusivity period …",
        "doc_tech": "The new transformer architecture achieves SOTA on MMLU …",
    }

    steps = [
        PipelineStep(
            name=f"summary_{key}",
            fn=make_summariser(llm, key),
            description=f"Summarise {key}",
        )
        for key in documents
    ]

    # A final aggregation step depends on all summaries
    def aggregate(ctx):
        summaries = [ctx.get(f"summary_{k}", "") for k in documents]
        return "\n".join(f"• {s}" for s in summaries)

    steps.append(
        PipelineStep(
            name="report",
            fn=aggregate,
            depends_on=[f"summary_{k}" for k in documents],
            description="Combine summaries into a report",
        )
    )

    pipeline = Pipeline(steps, name="enterprise_report")

    print("Running enterprise pipeline …\n")
    result = pipeline.run(documents)

    print("=== Individual Summaries ===")
    for key in documents:
        print(f"  [{key}] {result.get(f'summary_{key}')}")

    print("\n=== Final Report ===")
    print(result.get("report"))
    print(f"\nSteps run: {result.metadata['steps_run']}")


if __name__ == "__main__":
    main()
