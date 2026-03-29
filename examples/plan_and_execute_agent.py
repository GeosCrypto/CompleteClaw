"""Example: PlanAndExecuteAgent (AI Agent Plus) for multi-step tasks."""

import json

from completeclaw.agents.plan_execute import PlanAndExecuteAgent
from completeclaw.llm.mock import MockLLMProvider
from completeclaw.tools.calculator import CalculatorTool

# Simulate an LLM that:
#  1. Returns a JSON plan
#  2. Executes each step with optional tool use
#  3. Synthesises a final answer
responses = [
    # Planning phase: break the task into steps
    json.dumps([
        "Calculate the area of a circle with radius 5",
        "Calculate the circumference of a circle with radius 5",
        "Summarise the results",
    ]),
    # Execute step 1 – use the calculator
    'Thought: Area = pi * r^2\nAction: calculator\nAction Input: {"expression": "pi * 5 ** 2"}',
    "Final Answer: 78.53981633974483",
    # Execute step 2 – use the calculator
    'Thought: Circumference = 2 * pi * r\nAction: calculator\nAction Input: {"expression": "2 * pi * 5"}',
    "Final Answer: 31.41592653589793",
    # Execute step 3 – no tool needed
    "Final Answer: Area ≈ 78.54, Circumference ≈ 31.42",
    # Synthesis
    "Final Answer: For a circle with radius 5: Area ≈ 78.54 square units, Circumference ≈ 31.42 units.",
]

llm = MockLLMProvider(responses=responses)
agent = PlanAndExecuteAgent(llm, tools=[CalculatorTool()], max_plan_steps=5)

result = agent.run("Analyse a circle with radius 5: compute area and circumference")

print("=== AI Agent Plus – Plan and Execute ===")
print(f"\nFinal Answer: {result.output}")
print(f"\nPlan ({result.metadata['plan_steps']} steps):")
for i, step in enumerate(result.metadata["plan"], 1):
    print(f"  {i}. {step}")
print(f"\nExecution trace ({len(result.steps)} steps):")
for s in result.steps:
    print(f"  Step {s['plan_step']}: {s['step']}")
    print(f"    → {s['result']}")
