"""Example: Parallel pipeline with dependency resolution."""

from completeclaw.workflows.pipeline import Pipeline

pipeline = Pipeline(name="data_pipeline")
pipeline.add_step("ingest", fn=lambda ctx: {"records": [1, 2, 3, 4, 5]})
pipeline.add_step(
    "validate",
    fn=lambda ctx: [r for r in ctx["ingest"]["records"] if r > 0],
    depends_on=["ingest"],
)
pipeline.add_step(
    "transform",
    fn=lambda ctx: [r * 2 for r in ctx["validate"]],
    depends_on=["validate"],
)
pipeline.add_step(
    "report",
    fn=lambda ctx: f"Processed {len(ctx['transform'])} records: {ctx['transform']}",
    depends_on=["transform"],
)

result = pipeline.run()
print(result.get("report"))
