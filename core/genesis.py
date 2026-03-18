"""Genesis Agent — the top-level orchestrator that receives user goals and evolves.

Uses LangGraph for the agent runtime. The Genesis agent is itself a graph node
that can spawn child agent graphs via the spawn_agent tool.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
from langgraph.checkpoint.memory import MemorySaver
from motor.motor_asyncio import AsyncIOMotorClient

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_STEPS,
    DEFAULT_TTL_SECONDS,
    GENESIS_SPAWN_BUDGET,
    MAX_DEPTH,
    MAX_GLOBAL_AGENTS,
    MONGO_DB_NAME,
    MONGO_URI,
    REDIS_URL,
    SESSION_BUDGET_USD,
)
from core.agent import run_agent_graph
from core.blueprint import (
    AgentBlueprint,
    MemoryScope,
    SpawnBudget,
    compile_system_prompt,
)
from core.cost_tracker import CostTracker
from core.spawner import Spawner
from tracing.event_logger import EventLogger
from tracing.event_types import AGENT_SPAWNED
from memory.locks import SpawnLock
from memory.registry import AgentRegistry
from prompts.genesis import GENESIS_PERSONA

logger = logging.getLogger("evolve.genesis")


class GenesisAgent:
    """Top-level orchestrator. Receives a user goal, decomposes, spawns, synthesizes.

    The Genesis agent runs as a LangGraph StateGraph. When it calls the
    spawn_agent tool, child agents are themselves full LangGraph instances
    that run to completion and return their output as tool results.
    """

    def __init__(
        self,
        session_budget_usd: float = SESSION_BUDGET_USD,
        model: str = DEFAULT_LLM_MODEL,
    ):
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.session_budget_usd = session_budget_usd

        self._mongo_client: AsyncIOMotorClient | None = None
        self._redis: aioredis.Redis | None = None
        self.registry: AgentRegistry | None = None
        self.event_logger: EventLogger | None = None
        self.cost_tracker: CostTracker | None = None
        self.spawner: Spawner | None = None
        self._checkpointer: MemorySaver | None = None

    async def _init_infrastructure(self) -> None:
        """Connect to MongoDB and Redis, ensure indexes."""
        logger.info("Connecting to MongoDB at %s", MONGO_URI)
        self._mongo_client = AsyncIOMotorClient(MONGO_URI)
        db = self._mongo_client[MONGO_DB_NAME]

        logger.info("Connecting to Redis at %s", REDIS_URL)
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)

        self.registry = AgentRegistry(db)
        await self.registry.ensure_indexes()

        self.event_logger = EventLogger(db)
        await self.event_logger.ensure_indexes()

        self.cost_tracker = CostTracker(
            registry=self.registry,
            event_logger=self.event_logger,
            session_id=self.session_id,
            budget_usd=self.session_budget_usd,
        )

        spawn_lock = SpawnLock(self._redis)
        self.spawner = Spawner(
            registry=self.registry,
            event_logger=self.event_logger,
            cost_tracker=self.cost_tracker,
            spawn_lock=spawn_lock,
            session_id=self.session_id,
        )

        # Shared checkpointer for all agents in this session
        self._checkpointer = MemorySaver()
        logger.info("Infrastructure initialized for session %s", self.session_id)

    async def run(self, user_goal: str) -> dict:
        """Execute the full Evolve pipeline for a user goal.

        Returns a dict with:
        - output: The final synthesized response
        - session_id: The session identifier
        - token_summary: Per-agent and total token usage
        - agents_spawned: Number of agents that were created
        - agent_tree: Hierarchical view of all agents
        """
        await self._init_infrastructure()

        try:
            logger.info("Session %s started | goal='%s' | model=%s | budget=$%.2f",
                        self.session_id, user_goal[:100], self.model, self.session_budget_usd)
            # 1. Create the Genesis agent blueprint
            genesis_blueprint = AgentBlueprint(
                name="Genesis-Orchestrator",
                role="orchestrator",
                persona=GENESIS_PERSONA,
                expertise=["decomposition", "orchestration", "synthesis"],
                tools_allowed=["spawn_agent", "web_search", "share_finding", "read_shared"],
                tools_denied=[],
                model=self.model,
                memory_scope=MemoryScope(
                    write_ns="agent:genesis",
                    read_ns=["shared"],
                    share_policy="full_transparency",
                ),
                ttl_seconds=DEFAULT_TTL_SECONDS * 2,
                max_steps=DEFAULT_MAX_STEPS * 2,
                priority="critical",
                spawn_budget=SpawnBudget(
                    max_children=GENESIS_SPAWN_BUDGET,
                    max_depth_remaining=MAX_DEPTH,
                    remaining_global=MAX_GLOBAL_AGENTS,
                ),
                task=user_goal,
                success_criteria="Fully address the user's goal by orchestrating specialist agents and synthesizing their outputs.",
                output_format="markdown",
                depth=0,
            )

            # Register genesis in MongoDB
            await self.registry.register(
                genesis_blueprint, session_id=self.session_id, parent_id=None
            )
            system_prompt = compile_system_prompt(genesis_blueprint)
            await self.registry.update(
                genesis_blueprint.agent_id, system_prompt_compiled=system_prompt
            )

            # Log genesis spawn
            await self.event_logger.log_event(
                session_id=self.session_id,
                agent_id=genesis_blueprint.agent_id,
                agent_name=genesis_blueprint.name,
                depth=0,
                event_type=AGENT_SPAWNED,
                payload={
                    "parent_id": None,
                    "role": "orchestrator",
                    "task": user_goal,
                    "model": self.model,
                    "spawn_reason": "user_request",
                },
            )

            # 2. Run the genesis agent as a LangGraph
            logger.info("Running Genesis agent graph (id=%s)", genesis_blueprint.agent_id)
            output = await run_agent_graph(
                blueprint=genesis_blueprint,
                session_id=self.session_id,
                cost_tracker=self.cost_tracker,
                event_logger=self.event_logger,
                blackboard_redis=self._redis,
                spawner=self.spawner,
                checkpointer=self._checkpointer,
            )

            # 3. Gather session summary
            token_summary = self.cost_tracker.get_session_summary()
            all_agents = await self.registry.get_session_agents(self.session_id)
            logger.info("Session %s complete | agents=%d | cost=$%.4f | tokens=%d",
                        self.session_id, len(all_agents),
                        token_summary["total_cost_usd"], token_summary["total_tokens"])

            return {
                "output": output,
                "session_id": self.session_id,
                "token_summary": token_summary,
                "agents_spawned": len(all_agents),
                "agent_tree": [
                    {
                        "id": a["_id"],
                        "name": a["name"],
                        "role": a["blueprint"]["role"],
                        "depth": a["depth"],
                        "status": a["status"],
                        "token_usage": a.get("token_usage", {}),
                        "subtree_token_usage": a.get("subtree_token_usage", {}),
                    }
                    for a in all_agents
                ],
            }

        finally:
            logger.info("Closing connections for session %s", self.session_id)
            if self._redis:
                await self._redis.aclose()
            if self._mongo_client:
                self._mongo_client.close()
