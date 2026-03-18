"""Redis distributed locks for spawn budget concurrency control."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis

from config import REDIS_URL


class SpawnLock:
    """Async Redis-based distributed lock for safe spawn budget enforcement."""

    def __init__(self, redis: aioredis.Redis | None = None):
        self.redis = redis or aioredis.from_url(REDIS_URL, decode_responses=True)

    @asynccontextmanager
    async def acquire(self, session_id: str, timeout: float = 5.0) -> AsyncGenerator[bool, None]:
        """Context manager that acquires a distributed lock for spawning.

        Usage:
            async with spawn_lock.acquire(session_id) as acquired:
                if acquired:
                    # safe to check + decrement spawn budget
        """
        lock = self.redis.lock(
            f"spawn_lock:{session_id}",
            timeout=timeout,
            blocking_timeout=3.0,
        )
        acquired = await lock.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                await lock.release()
