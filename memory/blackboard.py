"""Redis-backed Blackboard — KV shared memory with pub/sub and streams."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import redis.asyncio as aioredis

from config import BLACKBOARD_READ_TIMEOUT, REDIS_URL

logger = logging.getLogger("evolve.blackboard")


class BlackboardClient:
    """Per-agent client for Redis-backed shared memory, pub/sub, and streams."""

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        redis: aioredis.Redis | None = None,
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.redis = redis or aioredis.from_url(REDIS_URL, decode_responses=True)
        self._stream_key = f"stream:{session_id}"
        self._notify_channel = f"notify:{session_id}"

    # ── Key-Value (Blackboard) ──────────────────────────────────

    async def write(self, key: str, value: Any) -> None:
        """Write to agent's private namespace."""
        full_key = f"blackboard:{self.session_id}:agent:{self.agent_id}:{key}"
        await self.redis.set(full_key, json.dumps(value))

    async def share(self, key: str, value: Any) -> None:
        """Publish to shared namespace + notify listeners via pub/sub."""
        full_key = f"blackboard:{self.session_id}:shared:{key}"
        await self.redis.set(full_key, json.dumps(value))
        await self.redis.publish(
            self._notify_channel,
            json.dumps({"agent": self.agent_id, "key": key, "event": "shared"}),
        )

    async def read_shared(self, pattern: str = "*") -> dict[str, Any]:
        """Read all shared findings matching a pattern."""
        full_pattern = f"blackboard:{self.session_id}:shared:{pattern}"
        keys = []
        async for key in self.redis.scan_iter(match=full_pattern):
            keys.append(key)
        result = {}
        for k in keys:
            raw = await self.redis.get(k)
            if raw:
                result[k] = json.loads(raw)
        return result

    async def read_mine(self, pattern: str = "*") -> dict[str, Any]:
        """Read own private memory."""
        full_pattern = f"blackboard:{self.session_id}:agent:{self.agent_id}:{pattern}"
        keys = []
        async for key in self.redis.scan_iter(match=full_pattern):
            keys.append(key)
        result = {}
        for k in keys:
            raw = await self.redis.get(k)
            if raw:
                result[k] = json.loads(raw)
        return result

    # ── Pub/Sub (Push Notifications) ────────────────────────────

    async def wait_for_key(self, key: str, timeout: int = BLACKBOARD_READ_TIMEOUT) -> Any | None:
        """Block until a specific shared key appears. Push-based, no polling."""
        # Check if already available
        full_key = f"blackboard:{self.session_id}:shared:{key}"
        existing = await self.redis.get(full_key)
        if existing:
            return json.loads(existing)

        # Subscribe and wait
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._notify_channel)
        deadline = time.monotonic() + timeout
        try:
            async for message in pubsub.listen():
                if time.monotonic() > deadline:
                    return None
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    if data.get("key") == key:
                        raw = await self.redis.get(full_key)
                        return json.loads(raw) if raw else None
        finally:
            await pubsub.unsubscribe(self._notify_channel)
            await pubsub.aclose()

    # ── Streams (Task Dispatch) ─────────────────────────────────

    async def dispatch_task(self, target_agent_id: str, task: dict) -> None:
        """Send a task to a specific agent via Redis Stream."""
        await self.redis.xadd(
            self._stream_key,
            {
                "from": self.agent_id,
                "to": target_agent_id,
                "type": "task",
                "payload": json.dumps(task),
            },
        )

    async def consume_tasks(self) -> list[dict]:
        """Read tasks dispatched to this agent from the stream."""
        group = f"group:{self.agent_id}"
        try:
            await self.redis.xgroup_create(self._stream_key, group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists

        messages = await self.redis.xreadgroup(
            group, self.agent_id, {self._stream_key: ">"}, count=10
        )
        results = []
        if messages:
            for _stream, entries in messages:
                for _msg_id, fields in entries:
                    if fields.get("to") == self.agent_id:
                        results.append(json.loads(fields["payload"]))
        return results

    # ── Cleanup ─────────────────────────────────────────────────

    async def cleanup_session(self, session_id: str) -> None:
        """Remove all blackboard keys for a completed session."""
        pattern = f"blackboard:{session_id}:*"
        keys = []
        async for key in self.redis.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await self.redis.delete(*keys)
