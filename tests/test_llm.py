"""Tests for LLM provider abstractions."""

from __future__ import annotations

from completeclaw.llm.base import LLMResponse, Message, Role
from completeclaw.llm.mock import MockLLMProvider

# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------


class TestMockLLMProvider:
    def test_single_response(self):
        llm = MockLLMProvider(responses=["Hello!"])
        resp = llm.chat([Message(role=Role.USER, content="Hi")])
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello!"

    def test_multiple_responses_cycle(self):
        llm = MockLLMProvider(responses=["first", "second"])
        r1 = llm.chat([Message(role=Role.USER, content="1")])
        r2 = llm.chat([Message(role=Role.USER, content="2")])
        r3 = llm.chat([Message(role=Role.USER, content="3")])  # repeats last
        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "second"

    def test_reply_fn_overrides_responses(self):
        llm = MockLLMProvider(
            responses=["ignored"],
            reply_fn=lambda msgs: f"echo:{msgs[-1].content}",
        )
        resp = llm.chat([Message(role=Role.USER, content="test")])
        assert resp.content == "echo:test"

    def test_complete_helper(self):
        llm = MockLLMProvider(responses=["42"])
        result = llm.complete("What is 6 * 7?")
        assert result == "42"

    def test_complete_with_system(self):
        def _reply(msgs):
            return f"roles:{','.join(m.role.value for m in msgs)}"

        llm = MockLLMProvider(reply_fn=_reply)
        result = llm.complete("q", system="sys")
        assert result == "roles:system,user"

    def test_stream_default(self):
        llm = MockLLMProvider(responses=["streamed"])
        tokens = list(llm.stream([Message(role=Role.USER, content="go")]))
        assert tokens == ["streamed"]

    def test_reset(self):
        llm = MockLLMProvider(responses=["a", "b"])
        llm.chat([Message(role=Role.USER, content="1")])
        llm.reset()
        resp = llm.chat([Message(role=Role.USER, content="2")])
        assert resp.content == "a"

    def test_model_reported(self):
        llm = MockLLMProvider(responses=["ok"], model="my-model")
        resp = llm.chat([Message(role=Role.USER, content="hi")])
        assert resp.model == "my-model"

    def test_model_override_in_chat(self):
        llm = MockLLMProvider(responses=["ok"], model="default-model")
        resp = llm.chat([Message(role=Role.USER, content="hi")], model="override-model")
        assert resp.model == "override-model"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_dict_basic(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_to_dict_with_name(self):
        msg = Message(role=Role.TOOL, content="result", name="calculator")
        d = msg.to_dict()
        assert d["name"] == "calculator"

    def test_str(self):
        resp = LLMResponse(content="hi", model="m")
        assert str(resp) == "hi"
