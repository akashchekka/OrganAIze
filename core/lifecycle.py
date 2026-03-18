"""Agent lifecycle manager — TTL, heartbeat, hibernate, resurrect."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict

import redis.asyncio as aioredis

from core.blueprint import AgentBlueprint, compile_system_prompt
from tracing.event_logger import EventLogger
from prompts.lifecycle import RESURRECTION_CONTEXT_TEMPLATE
from tracing.event_types import (
    AGENT_DIED,
    AGENT_HIBERNATED,
    AGENT_RESURRECTED,
    AGENT_TTL_EXTENDED,
)
from memory.registry import AgentRegistry

logger = logging.getLogger("evolve.lifecycle")


class AgentExpired(Exception):
    """Raised when an agent's TTL has expired."""


class AgentLifecycle:
    """Manages TTL heartbeats, hibernation, and resurrection for a single agent."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        depth: int,
        ttl_seconds: int,
        max_steps: int,
        session_id: str,
        registry: AgentRegistry,
        event_logger: EventLogger,
        redis: aioredis.Redis,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.depth = depth
        self.ttl_seconds = ttl_seconds
        self.max_steps = max_steps
        self.session_id = session_id
        self.registry = registry
        self.event_logger = event_logger
        self.redis = redis

    async def start(self) -> None:
        """Initialize the heartbeat key in Redis."""
        logger.debug("Starting heartbeat for %s (ttl=%ds)", self.agent_name, self.ttl_seconds)
        await self.redis.setex(
            f"heartbeat:{self.agent_id}", self.ttl_seconds, "alive"
        )

    async def check_alive(self) -> bool:
        """Check if the agent's heartbeat is still valid."""
        return bool(await self.redis.exists(f"heartbeat:{self.agent_id}"))

    async def step(self) -> int:
        """Called after each agent action. Checks TTL and step limit.
        Returns updated steps_taken.
        """
        if not await self.check_alive():
            await self.hibernate(reason="ttl_expired")
            raise AgentExpired(f"Agent {self.agent_id} TTL expired")

        steps = await self.registry.increment_steps(self.agent_id)
        if steps >= self.max_steps:
            await self.die(reason="max_steps_exceeded")
            raise AgentExpired(f"Agent {self.agent_id} exceeded max steps ({self.max_steps})")
        return steps

    async def extend_ttl(self, extra_seconds: int) -> None:
        """Extend the agent's TTL if it's making progress."""
        remaining = await self.redis.ttl(f"heartbeat:{self.agent_id}")
        if remaining > 0:
            await self.redis.expire(
                f"heartbeat:{self.agent_id}", remaining + extra_seconds
            )
            await self.event_logger.log_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                depth=self.depth,
                event_type=AGENT_TTL_EXTENDED,
                payload={"extra_seconds": extra_seconds, "new_remaining": remaining + extra_seconds},
            )

    async def hibernate(self, reason: str = "ttl_expired", saved_state: dict | None = None) -> None:
        """Put the agent to sleep — preserves state for resurrection."""
        logger.info("Hibernating agent %s (reason=%s)", self.agent_name, reason)
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(
            self.agent_id,
            status="hibernated",
            exit_reason=reason,
            saved_state=saved_state or {},
        )
        await self.event_logger.log_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            depth=self.depth,
            event_type=AGENT_HIBERNATED,
            payload={"reason": reason},
        )

    async def complete(self, final_output: str) -> None:
        """Mark the agent as successfully completed."""
        logger.debug("Agent %s completed", self.agent_name)
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(
            self.agent_id,
            status="completed",
            exit_reason="completed",
            final_output=final_output,
        )

    async def die(self, reason: str = "killed") -> None:
        """Terminate the agent."""
        logger.info("Agent %s died (reason=%s)", self.agent_name, reason)
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(
            self.agent_id,
            status="dead",
            exit_reason=reason,
        )
        await self.event_logger.log_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            depth=self.depth,
            event_type=AGENT_DIED,
            payload={"reason": reason},
        )


async def resurrect(
    agent_id: str,
    registry: AgentRegistry,
    event_logger: EventLogger,
    redis: aioredis.Redis,
    session_id: str,
    new_task: str | None = None,
    new_ttl: int | None = None,
) -> tuple[AgentBlueprint, str, dict]:
    """Resurrect a hibernated agent. Returns (blueprint, system_prompt, saved_state)."""
    record = await registry.find_by_id(agent_id)
    if not record or record["status"] != "hibernated":
        raise ValueError(f"Agent {agent_id} is not hibernated (status={record.get('status') if record else 'not found'})")

    bp_data = record["blueprint"]
    saved_state = record.get("saved_state", {})

    blueprint = AgentBlueprint(**{
        k: v for k, v in bp_data.items()
        if k in AgentBlueprint.__dataclass_fields__
    })

    if new_task:
        blueprint.task = new_task

    ttl = new_ttl or blueprint.ttl_seconds
    system_prompt = compile_system_prompt(blueprint)

    # Add context about previous state
    resume_instruction = f"New additional task: {new_task}" if new_task else "Resume where you left off."
    resurrection_context = RESURRECTION_CONTEXT_TEMPLATE.format(
        partial_results=saved_state.get('partial_results', 'None'),
        last_step=saved_state.get('last_step', 'None'),
        resume_instruction=resume_instruction,
    )

    # Update registry
    await registry.update(
        agent_id,
        status="alive",
        ttl_seconds=ttl,
        expires_at=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=ttl)).isoformat(),
        exit_reason=None,
    )

    # Set heartbeat
    await redis.setex(f"heartbeat:{agent_id}", ttl, "alive")

    # Log
    await event_logger.log_event(
        session_id=session_id,
        agent_id=agent_id,
        agent_name=blueprint.name,
        depth=blueprint.depth,
        event_type=AGENT_RESURRECTED,
        payload={"new_task": new_task, "previous_state_keys": list(saved_state.keys())},
    )

    return blueprint, system_prompt + resurrection_context, saved_state
