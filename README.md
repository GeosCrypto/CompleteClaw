# CompleteClaw 🦾

> **The AI Swiss Army Knife** – a modular, extensible, and future-proof Python framework for all AI workflows, from local LLMs to enterprise-grade multi-sector agents.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why CompleteClaw?

| Feature | Detail |
|---------|--------|
| **Zero mandatory dependencies** | Core framework is pure stdlib – no `pip install` surprise |
| **Plug-and-play LLM backends** | OpenAI, Anthropic, Ollama, or mock (offline testing) |
| **Three agent patterns** | `ReActAgent` · `PlanAndExecuteAgent` · `RouterAgent` (multi-agent delegation) |
| **Extensible tool system** | Calculator, File I/O, HTTP requests, JSON, Web search – or build your own |
| **Flexible memory** | Sliding-window buffer, LLM-compressed summary, or persistent file-backed memory |
| **Workflow orchestration** | Sequential chains, parallel pipelines, and conditional if/else branching |
| **Retry utility** | Exponential-backoff decorator for transient LLM / API failures |

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

# Development
pip install "completeclaw[dev]"
```

---

## Quick Start

```python
from completeclaw import MockLLMProvider, ReActAgent, CalculatorTool

# Mock LLM: first call uses the calculator, second gives the final answer
llm = MockLLMProvider(responses=[
    'Thought: I need to calculate.\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
    "Final Answer: 42",
])
agent = ReActAgent(llm, tools=[CalculatorTool()])
result = agent.run("What is 6 × 7?")
print(result.output)          # "42"
print(result.steps[0]["observation"])  # "42"  (returned by CalculatorTool)
```

---

## Architecture

```
completeclaw/
├── llm/          LLM provider abstraction (OpenAI · Anthropic · Ollama · Mock)
├── agents/       Agent patterns (ReActAgent · PlanAndExecuteAgent · RouterAgent)
├── tools/        Tool/plugin system (Calculator · FileIO · HTTP · JSON · Search)
├── memory/       Memory stores (BufferMemory · SummaryMemory · FileMemory)
├── workflows/    Workflow orchestration (SequentialChain · Pipeline · ConditionalChain)
└── utils/        Helpers (structured logging · retry decorator)
```

---

## Modules

### LLM Providers (`completeclaw.llm`)

All providers implement the same `LLMProvider` interface:

```python
from completeclaw.llm.openai import OpenAIProvider
from completeclaw.llm.anthropic import AnthropicProvider
from completeclaw.llm.ollama import OllamaProvider
from completeclaw.llm.mock import MockLLMProvider   # no API key needed

# OpenAI (reads OPENAI_API_KEY env var)
llm = OpenAIProvider(default_model="gpt-4o-mini")

# Anthropic (reads ANTHROPIC_API_KEY env var)
llm = AnthropicProvider(default_model="claude-3-5-haiku-latest")

# Local Ollama
llm = OllamaProvider(base_url="http://localhost:11434", default_model="llama3")

# Mock – for tests & offline development
llm = MockLLMProvider(responses=["Hello!", "How can I help?"])
# or with a dynamic reply function:
llm = MockLLMProvider(reply_fn=lambda msgs: f"Echo: {msgs[-1].content}")
```

**Streaming** (all providers):
```python
for token in llm.stream(messages):
    print(token, end="", flush=True)
```

---

### Agents (`completeclaw.agents`)

#### `ReActAgent` – Reason + Act

Iterates through **Thought → Action → Observation** cycles until a final answer
is reached or `max_iterations` is exceeded.

```python
from completeclaw import ReActAgent, CalculatorTool, OpenAIProvider

llm = OpenAIProvider()
agent = ReActAgent(
    llm,
    tools=[CalculatorTool()],
    max_iterations=10,
    system_prompt="Be concise.",
)
result = agent.run("What is sqrt(1764)?")
print(result.output)          # "42"
print(result.steps)           # list of Thought/Action/Observation dicts
print(result.metadata)        # {"iterations": 2}
```

#### `PlanAndExecuteAgent` – AI Agent Plus

Designed for **complex, multi-step, multi-sector tasks**. Operates in three phases:

1. **Plan** – decomposes the task into an ordered JSON list of steps
2. **Execute** – runs each step through a ReAct-style loop (tool calls supported)
3. **Synthesise** – consolidates step results into a single coherent answer

```python
from completeclaw import PlanAndExecuteAgent, CalculatorTool, HttpRequestTool
from completeclaw.llm.openai import OpenAIProvider

agent = PlanAndExecuteAgent(
    llm=OpenAIProvider(),
    tools=[CalculatorTool(), HttpRequestTool()],
    max_plan_steps=8,
    max_iterations=5,
)
result = agent.run("Research the current Bitcoin price and convert it to EUR at 0.93 rate")
print(result.output)
print(result.metadata["plan"])   # ["Fetch BTC/USD price", "Convert to EUR", "Format answer"]
```

#### Build your own agent

```python
from completeclaw import Agent, AgentResult

class MyAgent(Agent):
    def run(self, task: str, **kwargs) -> AgentResult:
        reply = self.llm.complete(task)
        return AgentResult(output=reply)
```

#### `RouterAgent` – Multi-agent delegation

Routes each incoming task to the most appropriate specialised sub-agent.  The
routing decision is made by the LLM, which is prompted with the list of agents
and their descriptions.

```python
from completeclaw import RouterAgent, ReActAgent, CalculatorTool
from completeclaw.llm.openai import OpenAIProvider

math_agent = ReActAgent(OpenAIProvider(), tools=[CalculatorTool()])
chat_agent = ReActAgent(OpenAIProvider())

router = RouterAgent(
    llm=OpenAIProvider(),
    agents={"math": math_agent, "chat": chat_agent},
    descriptions={"math": "Handles calculations and maths", "chat": "General conversation"},
    fallback=chat_agent,   # used when routing cannot decide
)
result = router.run("What is sqrt(1764)?")
print(result.output)                    # "42"
print(result.metadata["routed_to"])     # "math"
```

Register sub-agents at runtime:

```python
router.register("code", code_agent, description="Writes and explains Python code")
```

---

### Tools (`completeclaw.tools`)

#### `CalculatorTool`

Safe AST-based math evaluator – supports `+`, `-`, `*`, `/`, `**`, `%`, `//`,
`sqrt`, `sin`, `cos`, `tan`, `log`, `pi`, `e`, and more.

```python
from completeclaw import CalculatorTool

tool = CalculatorTool()
print(tool.run(expression="sqrt(1764)").output)   # "42"
print(tool.run(expression="2 ** 10").output)      # "1024"
print(tool.run(expression="pi * 5 ** 2").output)  # "78.53981633974483"
```

#### `FileReadTool` / `FileWriteTool`

```python
from completeclaw import FileReadTool, FileWriteTool

FileWriteTool().run(path="/tmp/note.txt", content="hello world")
print(FileReadTool().run(path="/tmp/note.txt").output)  # "hello world"
```

#### `HttpRequestTool`

Makes HTTP/S requests using only the standard library (no `requests` needed).

```python
from completeclaw import HttpRequestTool

tool = HttpRequestTool()

# GET request
result = tool.run(url="https://api.github.com/repos/python/cpython")
print(result.metadata["status"])   # 200

# POST with JSON body
result = tool.run(
    url="https://httpbin.org/post",
    method="POST",
    body={"key": "value"},
    headers={"Authorization": "Bearer <token>"},
)
```

#### `WebSearchTool`

Stub search tool (swap in `TavilySearchTool` for live results):

```python
from completeclaw.tools.search import TavilySearchTool   # requires TAVILY_API_KEY

tool = TavilySearchTool()
result = tool.run(query="Python AI frameworks 2025")
print(result.output)
```

#### `JsonTool`

Parses, queries, and formats JSON data using only the standard library.
Ideal for extracting fields from API responses.

```python
from completeclaw import JsonTool

tool = JsonTool()

# Pretty-print / parse
r = tool.run(data='{"name":"Alice","scores":[95,87,92]}')
print(r.output)
# {
#   "name": "Alice",
#   "scores": [95, 87, 92]
# }

# Query a nested value (dot-notation; integers index into lists)
r = tool.run(operation="query", data='{"name":"Alice","scores":[95,87,92]}', path="scores.1")
print(r.output)  # "87"

# Query returns a sub-object as pretty JSON
r = tool.run(operation="query", data='{"user":{"name":"Bob","age":30}}', path="user")
print(r.output)
# {
#   "name": "Bob",
#   "age": 30
# }
```

#### Build your own tool

```python
from completeclaw import Tool, ToolResult

class TimeTool(Tool):
    name = "current_time"
    description = "Returns the current UTC time."

    def run(self, **kwargs) -> ToolResult:
        from datetime import datetime, timezone
        return ToolResult(output=datetime.now(timezone.utc).isoformat())
```

---

### Memory (`completeclaw.memory`)

#### `BufferMemory` – sliding-window cache

```python
from completeclaw import BufferMemory, ReActAgent

mem = BufferMemory(max_entries=50)
agent = ReActAgent(llm, memory=mem)

agent.run("My name is Alice.")
agent.run("What is my name?")   # agent recalls "Alice" from memory
```

#### `SummaryMemory` – LLM-compressed history

When the buffer exceeds `max_entries` the oldest entries are summarised by the
LLM, keeping the context window bounded.

```python
from completeclaw import SummaryMemory
from completeclaw.llm.openai import OpenAIProvider

mem = SummaryMemory(llm=OpenAIProvider(), max_entries=20)
```

#### `FileMemory` – persistent file-backed memory

Saves all entries to a JSON file immediately on every write.  History survives
process restarts and crashes.

```python
from completeclaw import FileMemory, ReActAgent
from completeclaw.llm.openai import OpenAIProvider

# First session
mem = FileMemory(path="~/.completeclaw/history.json", max_entries=100)
agent = ReActAgent(OpenAIProvider(), memory=mem)
agent.run("My name is Alice.")

# Later session – history is automatically reloaded from disk
mem2 = FileMemory(path="~/.completeclaw/history.json")
agent2 = ReActAgent(OpenAIProvider(), memory=mem2)
result = agent2.run("What is my name?")   # recalls "Alice"
```

---

### Workflows (`completeclaw.workflows`)

#### `SequentialChain`

Executes steps one after another; each step receives the shared context dict
(which accumulates all previous outputs).

```python
from completeclaw import SequentialChain, WorkflowStep

chain = SequentialChain(steps=[
    WorkflowStep("clean",  fn=lambda ctx: ctx["raw"].strip().lower()),
    WorkflowStep("greet",  fn=lambda ctx: f"Hello, {ctx['clean']}!"),
])
result = chain.run(context={"raw": "  World  "})
print(result.get("greet"))   # "Hello, world!"
```

#### `Pipeline`

Runs independent steps concurrently using threads; respects `depends_on`
for ordering.

```python
from completeclaw import Pipeline

p = Pipeline()
p.add_step("fetch",     fn=lambda ctx: fetch_data())
p.add_step("validate",  fn=lambda ctx: validate(ctx["fetch"]),  depends_on=["fetch"])
p.add_step("transform", fn=lambda ctx: transform(ctx["validate"]), depends_on=["validate"])
p.add_step("report",    fn=lambda ctx: summarise(ctx["transform"]), depends_on=["transform"])

result = p.run()
print(result.get("report"))
```

#### `ConditionalChain` – if/else branching

Extends `SequentialChain` with conditional guards and explicit if/else branch
points.  Steps skipped by their condition leave no key in the context.
Branches may be nested arbitrarily.

```python
from completeclaw import ConditionalChain, ConditionalStep, BranchStep

chain = ConditionalChain(steps=[
    ConditionalStep("score", fn=lambda ctx: int(ctx["raw_score"])),
    BranchStep(
        name="passed",
        condition=lambda ctx: ctx["score"] >= 60,
        if_steps=[
            ConditionalStep("grade", fn=lambda ctx: "Pass"),
            ConditionalStep("note",  fn=lambda ctx: "Well done!"),
        ],
        else_steps=[
            ConditionalStep("grade", fn=lambda ctx: "Fail"),
            ConditionalStep(
                "note",
                fn=lambda ctx: "Consider resitting.",
                condition=lambda ctx: ctx["score"] < 40,  # only below 40
            ),
        ],
    ),
])

r = chain.run(context={"raw_score": "75"})
print(r.get("grade"))   # "Pass"
print(r.get("passed"))  # True
print(r.get("note"))    # "Well done!"

r2 = chain.run(context={"raw_score": "35"})
print(r2.get("grade"))  # "Fail"
print(r2.get("note"))   # "Consider resitting."
```

---

## Utilities (`completeclaw.utils`)

### `retry` – exponential-backoff decorator

Retries a function on transient failures with configurable back-off and jitter.

```python
from completeclaw import retry

@retry(exceptions=ConnectionError, max_attempts=5, base_delay=1.0, backoff=2.0)
def call_llm(prompt: str) -> str:
    ...  # may raise ConnectionError transiently

# Use with an LLM provider method:
from completeclaw.utils.retry import retry as _retry

class MyProvider(LLMProvider):
    @_retry(exceptions=(RateLimitError,), max_attempts=4, base_delay=2.0)
    def chat(self, messages, **kwargs):
        ...
```

### `get_logger` – structured logging

```python
from completeclaw import get_logger

logger = get_logger(__name__)
logger.info("Agent started")
```

---

## Running Tests

```bash
pip install "completeclaw[dev]"
python -m pytest tests/ -v
```

## Linting

```bash
python -m ruff check completeclaw/ tests/
```

---

## Examples

| File | Description |
|------|-------------|
| `examples/react_agent.py` | ReActAgent with calculator + file I/O |
| `examples/plan_and_execute_agent.py` | PlanAndExecuteAgent (AI Agent Plus) |
| `examples/enterprise_pipeline.py` | Parallel pipeline with dependency resolution |

---

## License

MIT © CompleteClaw contributors

