"""
Example: ReAct agent with calculator and file I/O tools.

Run:
    OPENAI_API_KEY=sk-... python examples/react_agent.py

Or use the built-in MockLLMProvider to test without an API key:
    python examples/react_agent.py --mock
"""

from __future__ import annotations

import sys


def build_agent(use_mock: bool = False):
    from completeclaw.tools.calculator import CalculatorTool
    from completeclaw.tools.file_io import FileReadTool, FileWriteTool

    tools = [CalculatorTool(), FileReadTool(), FileWriteTool()]

    from completeclaw.agents.react import ReActAgent

    if use_mock:
        from completeclaw.llm.mock import MockLLMProvider

        responses = [
            'Thought: I can compute this directly.\nAction: calculator\nAction Input: {"expression": "sqrt(144) + 2 ** 8"}',
            "Final Answer: The result is 268.",
        ]
        llm = MockLLMProvider(responses=responses)
    else:
        from completeclaw.llm.openai import OpenAIProvider

        llm = OpenAIProvider()

    return ReActAgent(llm, tools=tools)


def main() -> None:
    use_mock = "--mock" in sys.argv

    print("Building ReAct agent …")
    agent = build_agent(use_mock=use_mock)
    print(agent)
    print()

    task = "What is sqrt(144) + 2 ** 8?"
    print(f"Task: {task}")
    result = agent.run(task)
    print(f"\nAnswer: {result.output}")
    print(f"Steps taken: {len(result.steps)}")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i}: [{step['action']}] → {step['observation']}")


if __name__ == "__main__":
    main()
