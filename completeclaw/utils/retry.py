"""Exponential-backoff retry decorator."""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

_F = TypeVar("_F", bound=Callable)  # type: ignore[type-arg]


def retry(
    *,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    jitter: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Callable[[_F], _F]:
    """Decorator that retries a callable on failure with exponential back-off.

    Parameters
    ----------
    exceptions:
        Exception type(s) to catch and retry on.  All other exceptions
        propagate immediately.
    max_attempts:
        Total number of attempts (first try plus retries).  Must be ≥ 1.
    base_delay:
        Initial wait in seconds before the first retry.
    max_delay:
        Maximum wait in seconds between retries (clamps exponential growth).
    backoff:
        Multiplier applied to the delay after each failed attempt.
    jitter:
        When ``True``, a small random amount (≤ 10 % of the current delay)
        is added to prevent thundering-herd effects.
    logger:
        If provided, a ``WARNING`` is logged before each retry with the
        exception message and wait time.

    Raises
    ------
    The last caught exception if all *max_attempts* are exhausted.

    Example::

        from completeclaw.utils.retry import retry

        @retry(exceptions=ConnectionError, max_attempts=5, base_delay=0.5)
        def call_api(prompt: str) -> str:
            ...  # may raise ConnectionError transiently

    Usage with an LLM provider::

        from completeclaw.utils.retry import retry

        class MyProvider(LLMProvider):
            @retry(exceptions=(RateLimitError,), max_attempts=4, base_delay=2.0)
            def chat(self, messages, **kwargs):
                ...
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            delay = base_delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    wait = min(delay, max_delay)
                    if jitter:
                        wait += random.uniform(0.0, wait * 0.1)
                    if logger is not None:
                        logger.warning(
                            "Attempt %d/%d failed (%s). Retrying in %.2fs.",
                            attempt,
                            max_attempts,
                            exc,
                            wait,
                        )
                    time.sleep(wait)
                    delay = min(delay * backoff, max_delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
