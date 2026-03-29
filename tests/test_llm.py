"""Tests for LLM providers."""

from __future__ import annotations

from completeclaw.llm.base import LLMResponse, Message, Role
from completeclaw.llm.mock import MockLLMProvider


class TestMessage:
    def test_to_dict_user(self):
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_dict_with_name(self):
        msg = Message(role=Role.TOOL, content="result", name="calc")
        d = msg.to_dict()
        assert d["name"] == "calc"

    def test_to_dict_no_name_key_when_absent(self):
        msg = Message(role=Role.USER, content="Hi")
        assert "name" not in msg.to_dict()


class TestLLMResponse:
    def test_str(self):
        r = LLMResponse(content="hello", model="gpt-4")
        assert str(r) == "hello"


class TestMockLLMProvider:
    def test_returns_predefined_responses_in_order(self):
        llm = MockLLMProvider(responses=["first", "second"])
        r1 = llm.chat([Message(role=Role.USER, content="a")])
        r2 = llm.chat([Message(role=Role.USER, content="b")])
        assert r1.content == "first"
        assert r2.content == "second"

    def test_repeats_last_response_when_exhausted(self):
        llm = MockLLMProvider(responses=["only"])
        llm.chat([Message(role=Role.USER, content="x")])
        r2 = llm.chat([Message(role=Role.USER, content="y")])
        assert r2.content == "only"

    def test_reply_fn_takes_priority(self):
        llm = MockLLMProvider(
            responses=["ignored"],
            reply_fn=lambda msgs: f"echo:{msgs[-1].content}",
        )
        r = llm.chat([Message(role=Role.USER, content="hi")])
        assert r.content == "echo:hi"

    def test_complete_helper(self):
        llm = MockLLMProvider(responses=["world"])
        assert llm.complete("hello") == "world"

    def test_stream_yields_content(self):
        llm = MockLLMProvider(responses=["streamed"])
        tokens = list(llm.stream([Message(role=Role.USER, content="go")]))
        assert "".join(tokens) == "streamed"

    def test_reset_clears_counter(self):
        llm = MockLLMProvider(responses=["a", "b"])
        llm.chat([Message(role=Role.USER, content="x")])
        llm.reset()
        r = llm.chat([Message(role=Role.USER, content="x")])
        assert r.content == "a"

    def test_default_response_when_no_responses_provided(self):
        llm = MockLLMProvider()
        r = llm.chat([Message(role=Role.USER, content="?")])
        assert isinstance(r.content, str)
        assert len(r.content) > 0
