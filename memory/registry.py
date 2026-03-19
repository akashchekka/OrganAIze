"""MongoDB-backed Agent Registry — stores blueprints, status, tokens, lineage."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import MONGO_DB_NAME, MONGO_URI
from core.blueprint import AgentBlueprint

logger = logging.getLogger("evolve.registry")


class AgentRegistry:
    """Async MongoDB client for the agents collection."""

    def __init__(self, db: AsyncIOMotorDatabase | None = None):
        if db is None:
            client = AsyncIOMotorClient(MONGO_URI)
            db = client[MONGO_DB_NAME]
        self.db = db
        self.agents = db["agents"]

    # ── Indexes (call once at startup) ──────────────────────────

    async def ensure_indexes(self):
        await self.agents.create_index("session_id")
        await self.agents.create_index("status")
        await self.agents.create_index([("session_id", 1), ("status", 1)])
        await self.agents.create_index("blueprint.expertise")
        await self.agents.create_index("lineage")

    # ── CRUD ────────────────────────────────────────────────────

    async def register(self, blueprint: AgentBlueprint, session_id: str, parent_id: str | None = None) -> dict:
        """Insert a new agent record from its blueprint."""
        doc = {
            "_id": blueprint.agent_id,
            "session_id": session_id,
            "name": blueprint.name,
            "lineage": blueprint.lineage,
            "parent_id": parent_id,
            "depth": blueprint.depth,
            "blueprint": asdict(blueprint),
            "system_prompt_compiled": None,  # set by caller
            "status": "alive",
            "created_at": dt.datetime.now(dt.timezone.utc),
            "ttl_seconds": blueprint.ttl_seconds,
            "expires_at": dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=blueprint.ttl_seconds),
            "steps_taken": 0,
            "max_steps": blueprint.max_steps,
            "children": [],
            "spawn_budget_remaining": blueprint.spawn_budget.max_children,
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
            },
            "subtree_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
                "agent_count": 1,
            },
            "final_output": None,
            "exit_reason": None,
            "saved_state": None,
        }
        await self.agents.insert_one(doc)
        return doc

    async def find_by_id(self, agent_id: str) -> dict | None:
        return await self.agents.find_one({"_id": agent_id})

    async def update(self, agent_id: str, **fields) -> None:
        """Update arbitrary fields on an agent record."""
        await self.agents.update_one({"_id": agent_id}, {"$set": fields})

    async def add_child(self, parent_id: str, child_id: str) -> None:
        await self.agents.update_one(
            {"_id": parent_id},
            {"$push": {"children": child_id}},
        )

    async def increment_steps(self, agent_id: str) -> int:
        """Increment steps_taken and return the new value."""
        result = await self.agents.find_one_and_update(
            {"_id": agent_id},
            {"$inc": {"steps_taken": 1}},
            return_document=True,
        )
        return result["steps_taken"] if result else 0

    # ── Queries for Reuse-First ─────────────────────────────────

    async def find_agent(
        self,
        session_id: str,
        status: str,
        capabilities: list[str] | None = None,
    ) -> dict | None:
        """Find an agent by status and overlapping capabilities."""
        query: dict[str, Any] = {"session_id": session_id, "status": status}
        if capabilities:
            query["blueprint.expertise"] = {"$in": capabilities}
        return await self.agents.find_one(query)

    async def count_alive(self, session_id: str) -> int:
        return await self.agents.count_documents(
            {"session_id": session_id, "status": "alive"}
        )

    async def get_session_agents(self, session_id: str) -> list[dict]:
        cursor = self.agents.find({"session_id": session_id})
        return await cursor.to_list(length=None)
