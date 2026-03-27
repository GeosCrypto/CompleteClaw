"""
Example: Chat with a local Ollama LLM using CompleteClaw.

Prerequisites:
    pip install completeclaw
    ollama pull llama3   # or any other model

Run:
    python examples/local_llm_chat.py
"""

from __future__ import annotations

from completeclaw.llm.ollama import OllamaProvider


def main() -> None:
    llm = OllamaProvider(default_model="llama3")

    print("CompleteClaw – Local LLM Chat (type 'quit' to exit)\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        from completeclaw.llm.base import Message, Role

        history.append(Message(role=Role.USER, content=user_input))

        print("Assistant: ", end="", flush=True)
        reply_parts = []
        for token in llm.stream(history):
            print(token, end="", flush=True)
            reply_parts.append(token)
        print()

        history.append(Message(role=Role.ASSISTANT, content="".join(reply_parts)))


if __name__ == "__main__":
    main()
