"""Per-process async singleflight for deduplicating concurrent identical work."""
from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class _SingleFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, asyncio.Task[object]] = {}

    async def do(self, key: str, work: Callable[[], Awaitable[T]]) -> T:
        loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._tasks.get(key)
            if existing is None or existing.get_loop() is not loop:
                task: asyncio.Task[T] = loop.create_task(work())
                self._tasks[key] = task
                leader = True
            else:
                task = existing  # type: ignore[assignment]
                leader = False

        try:
            return await asyncio.shield(task)
        finally:
            if leader:
                with self._lock:
                    if self._tasks.get(key) is task:
                        self._tasks.pop(key, None)


_instance = _SingleFlight()


async def singleflight_do(key: str, work: Callable[[], Awaitable[T]]) -> T:
    return await _instance.do(key, work)
