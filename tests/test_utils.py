"""Tests for utility helpers."""

from __future__ import annotations

import pytest

from completeclaw.utils.retry import retry

# ---------------------------------------------------------------------------
# retry decorator
# ---------------------------------------------------------------------------


class TestRetry:
    def test_succeeds_on_first_attempt(self):
        calls = []

        @retry(max_attempts=3)
        def fn():
            calls.append(1)
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        calls = []

        @retry(exceptions=ValueError, max_attempts=3, base_delay=0, jitter=False)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "done"

        assert fn() == "done"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self):
        calls = []

        @retry(exceptions=RuntimeError, max_attempts=3, base_delay=0, jitter=False)
        def fn():
            calls.append(1)
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            fn()
        assert len(calls) == 3

    def test_non_matching_exception_propagates_immediately(self):
        calls = []

        @retry(exceptions=ValueError, max_attempts=5, base_delay=0, jitter=False)
        def fn():
            calls.append(1)
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            fn()
        assert len(calls) == 1  # no retries

    def test_multiple_exception_types(self):
        calls = []

        @retry(
            exceptions=(ValueError, ConnectionError),
            max_attempts=4,
            base_delay=0,
            jitter=False,
        )
        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("v")
            if len(calls) == 2:
                raise ConnectionError("c")
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 3

    def test_max_attempts_one_no_retry(self):
        calls = []

        @retry(exceptions=Exception, max_attempts=1, base_delay=0, jitter=False)
        def fn():
            calls.append(1)
            raise Exception("fail")

        with pytest.raises(Exception):
            fn()
        assert len(calls) == 1

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            retry(max_attempts=0)

    def test_preserves_function_name(self):
        @retry()
        def my_function():
            return "x"

        assert my_function.__name__ == "my_function"

    def test_passes_args_and_kwargs(self):
        @retry()
        def add(a, b=0):
            return a + b

        assert add(3, b=4) == 7
