# 🦾 CompleteClaw

> **The AI Swiss Army Knife** – modular, extensible, and future-proof for every AI workflow,  
> from a single local LLM call to a full enterprise multi-agent pipeline.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero mandatory dependencies](https://img.shields.io/badge/dependencies-zero%20mandatory-brightgreen.svg)](#installation)

---

## Why CompleteClaw?

Most AI toolkits force you to commit to one vendor, one paradigm, or one use-case.  
**CompleteClaw** is different:

| Feature | CompleteClaw |
|---|---|
| Swap LLM providers | ✅ OpenAI · Anthropic · Ollama · any custom provider |
| Local & cloud LLMs | ✅ Zero mandatory dependencies |
| Plug-in tools | ✅ Calculator · File I/O · Web Search · bring your own |
| Agent frameworks | ✅ ReAct agent built-in; extend `Agent` for any pattern |
| Memory systems | ✅ Buffer · Summary · any back-end |
| Workflow orchestration | ✅ Sequential chains · parallel pipelines |
| Enterprise-ready | ✅ Concurrent execution, dependency graphs |
| Fully testable | ✅ `MockLLMProvider` for 100% offline tests |

---

## Installation

```bash
# Core (zero dependencies)
pip install completeclaw

# With OpenAI support
pip install "completeclaw[openai]"

# With Anthropic support
pip install "completeclaw[anthropic]"

# Everything
pip install "completeclaw[all]"
```

---

## Quick Start

### 1. Chat with any LLM

```python
# OpenAI
from completeclaw.llm.openai import OpenAIProvider
llm = OpenAIProvider(api_key="sk-...")
print(llm.complete("What is the capital of France?"))

# Anthropic
from completeclaw.llm.anthropic import AnthropicProvider
llm = AnthropicProvider(api_key="sk-ant-...")
print(llm.complete("Explain quantum entanglement simply."))

# Local Ollama (no API key needed!)
from completeclaw.llm.ollama import OllamaProvider
llm = OllamaProvider(default_model="llama3")
print(llm.complete("Tell me a joke."))

# Offline / testing
from completeclaw.llm.mock import MockLLMProvider
llm = MockLLMProvider(responses=["Hello from mock!"])
print(llm.complete("Hi"))
```

### 2. Build a ReAct Agent with Tools

```python
from completeclaw.llm.openai import OpenAIProvider
from completeclaw.agents.react import ReActAgent
from completeclaw.tools.calculator import CalculatorTool
from completeclaw.tools.search import TavilySearchTool

llm = OpenAIProvider()
agent = ReActAgent(
    llm,
    tools=[
        CalculatorTool(),
        TavilySearchTool(api_key="tvly-..."),
    ],
    max_iterations=10,
)

result = agent.run("What is sqrt(144) + 2^8, and who invented calculus?")
print(result.output)
for step in result.steps:
    print(f"  [{step['action']}] → {step['observation']}")
```

### 3. Sequential Workflow Chain

```python
from completeclaw.workflows.chain import SequentialChain
from completeclaw.workflows.base import WorkflowStep

chain = SequentialChain([
    WorkflowStep("fetch",    lambda ctx: f"Document about {ctx['topic']}"),
    WorkflowStep("summarise", lambda ctx: ctx["fetch"][:50] + "…"),
    WorkflowStep("translate", lambda ctx: f"[ES] {ctx['summarise']}"),
])

result = chain.run({"topic": "quantum computing"})
print(result.get("translate"))
```

### 4. Parallel Enterprise Pipeline

```python
from completeclaw.workflows.pipeline import Pipeline, PipelineStep

pipeline = Pipeline([
    PipelineStep("analyse_finance", lambda ctx: analyse(ctx["finance_doc"])),
    PipelineStep("analyse_legal",   lambda ctx: analyse(ctx["legal_doc"])),
    PipelineStep(
        "final_report",
        lambda ctx: merge(ctx["analyse_finance"], ctx["analyse_legal"]),
        depends_on=["analyse_finance", "analyse_legal"],
    ),
])

result = pipeline.run({"finance_doc": "...", "legal_doc": "..."})
print(result.get("final_report"))
```

### 5. Persistent Memory

```python
from completeclaw.memory.buffer import BufferMemory
from completeclaw.agents.react import ReActAgent
from completeclaw.llm.openai import OpenAIProvider

memory = BufferMemory(max_entries=20)
agent  = ReActAgent(OpenAIProvider(), memory=memory)

agent.run("My name is Alice.")
agent.run("What is my name?")   # Agent remembers "Alice"
```

---

## Architecture

```
completeclaw/
├── llm/            # LLM provider abstraction
│   ├── base.py     #   LLMProvider, Message, Role, LLMResponse
│   ├── openai.py   #   OpenAI API (GPT-4o, o1, …)
│   ├── anthropic.py#   Anthropic API (Claude 3.x)
│   ├── ollama.py   #   Local Ollama (Llama 3, Mistral, …)
│   └── mock.py     #   Offline mock for tests
├── agents/
│   ├── base.py     #   Agent ABC + AgentResult
│   └── react.py    #   ReAct (Reason + Act) agent
├── tools/
│   ├── base.py     #   Tool ABC + ToolRegistry + ToolResult
│   ├── calculator.py   #   Safe AST-based math evaluator
│   ├── file_io.py  #   FileReadTool, FileWriteTool
│   └── search.py   #   WebSearchTool + TavilySearchTool
├── memory/
│   ├── base.py     #   Memory ABC + MemoryEntry
│   ├── buffer.py   #   Sliding-window BufferMemory
│   └── summary.py  #   LLM-compressed SummaryMemory
├── workflows/
│   ├── base.py     #   Workflow ABC + WorkflowResult + WorkflowStep
│   ├── chain.py    #   SequentialChain
│   └── pipeline.py #   Parallel Pipeline with dependency resolution
└── utils/
    └── logging.py  #   Structured logger
```

---

## Extending CompleteClaw

### Custom LLM Provider

```python
from completeclaw.llm.base import LLMProvider, LLMResponse, Message
from typing import Sequence

class MyCustomProvider(LLMProvider):
    name = "my_provider"

    def chat(self, messages: Sequence[Message], **kwargs) -> LLMResponse:
        # Call your custom endpoint here
        reply = call_my_api([m.to_dict() for m in messages])
        return LLMResponse(content=reply, model="my-model")
```

### Custom Tool

```python
from completeclaw.tools.base import Tool, ToolResult

class WeatherTool(Tool):
    name = "weather"
    description = 'Get current weather. Input: {"city": "<city name>"}.'

    def run(self, *, city: str = "", **kwargs) -> ToolResult:
        data = fetch_weather(city)  # your implementation
        return ToolResult(output=f"{city}: {data['temp']}°C, {data['condition']}")
```

### Custom Agent

```python
from completeclaw.agents.base import Agent, AgentResult

class ChainOfThoughtAgent(Agent):
    def run(self, task: str, **kwargs) -> AgentResult:
        prompt = f"Think step by step. Task: {task}"
        reply = self.llm.complete(prompt)
        return AgentResult(output=reply)
```

---

## Running the Examples

```bash
# Local LLM chat (requires Ollama)
python examples/local_llm_chat.py

# ReAct agent (mock mode, no API key needed)
python examples/react_agent.py --mock

# Enterprise parallel pipeline
python examples/enterprise_pipeline.py
```

---

## Running the Tests

```bash
pip install "completeclaw[dev]"
pytest
```

---

## Contributing

Contributions are very welcome!  Here are some ideas:

- 🔌 New LLM providers (Groq, Mistral, Google Gemini, …)
- 🛠️ New tools (code execution, database queries, email, …)
- 🧠 New memory backends (Redis, Pinecone, SQLite, …)
- 🔀 New workflow patterns (map-reduce, retry-with-feedback, …)
- 📚 More examples and documentation

Please open an issue to discuss before submitting a large PR.

---

## License

MIT © GeosCrypto
