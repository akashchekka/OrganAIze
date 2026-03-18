"""Immutable event journal — append-only MongoDB log of all agent actions."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from config import MONGO_DB_NAME, MONGO_URI


@dataclass
class AgentEvent:
    """A single event in the immutable journal."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    session_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    depth: int = 0
    event_type: str = ""
    payload: dict = field(default_factory=dict)
    parent_event_id: str | None = None


class EventLogger:
    """Async MongoDB-backed immutable event journal."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["events"]

    async def ensure_indexes(self):
        await self.collection.create_index("session_id")
        await self.collection.create_index("agent_id")
        await self.collection.create_index("event_type")
        await self.collection.create_index("timestamp")
        await self.collection.create_index(
            [("session_id", 1), ("agent_id", 1), ("timestamp", 1)]
        )

    async def log(self, event: AgentEvent) -> None:
        """Append an event to the journal. Never update or delete."""
        await self.collection.insert_one(asdict(event))

    async def log_event(
        self,
        session_id: str,
        agent_id: str,
        agent_name: str,
        depth: int,
        event_type: str,
        payload: dict,
        parent_event_id: str | None = None,
    ) -> AgentEvent:
        """Convenience: build and log an event in one call."""
        event = AgentEvent(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            depth=depth,
            event_type=event_type,
            payload=payload,
            parent_event_id=parent_event_id,
        )
        await self.log(event)
        return event

    # ── Queries ─────────────────────────────────────────────────

    async def get_agent_trace(self, agent_id: str) -> list[dict]:
        """Full chronological trace of one agent's life."""
        cursor = self.collection.find({"agent_id": agent_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_session_events(self, session_id: str) -> list[dict]:
        """Everything that happened in a session."""
        cursor = self.collection.find({"session_id": session_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_spawn_tree(self, session_id: str) -> list[dict]:
        """Get all spawn events for hierarchy reconstruction."""
        cursor = self.collection.find(
            {"session_id": session_id, "event_type": "agent.spawned"}
        ).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_cost_events(self, session_id: str) -> list[dict]:
        """Get all cost events for budget analysis."""
        cursor = self.collection.find(
            {"session_id": session_id, "event_type": "agent.cost"}
        ).sort("timestamp", 1)
        return await cursor.to_list(length=None)
