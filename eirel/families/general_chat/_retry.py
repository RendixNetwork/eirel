from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

_T = TypeVar("_T")

_MAX_ATTEMPTS = 3
_BASE_DELAYS = (0.2, 0.4)
_JITTER_MAX = 0.1


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.HTTPError)


async def retry_with_jitter(
    fn: Callable[[], Awaitable[_T]],
) -> _T:
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return await fn()
        except Exception as exc:
            if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                raise
            delay = _BASE_DELAYS[attempt] + random.uniform(0, _JITTER_MAX)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
