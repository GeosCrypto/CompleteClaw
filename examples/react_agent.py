"""Example: ReAct agent with calculator and file I/O."""

from completeclaw.agents.react import ReActAgent
from completeclaw.llm.mock import MockLLMProvider
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.file_io import FileWriteTool

responses = [
    'Thought: I need to calculate sqrt(144).\nAction: calculator\nAction Input: {"expression": "sqrt(144)"}',
    'Thought: Now I\'ll save the result.\nAction: file_write\nAction Input: {"path": "/tmp/result.txt", "content": "12"}',
    "Final Answer: Saved 12 to /tmp/result.txt",
]

llm = MockLLMProvider(responses=responses)
agent = ReActAgent(llm, tools=[CalculatorTool(), FileWriteTool()])
result = agent.run("Calculate sqrt(144) and save to /tmp/result.txt")
print(result.output)
print(f"Steps taken: {len(result.steps)}")
