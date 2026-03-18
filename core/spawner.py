"""Spawner — assessment, budget enforcement, reuse-first, and agent creation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import litellm

from config import (
    MAX_DEPTH,
    MAX_GLOBAL_AGENTS,
    SPAWN_AUTO_THRESHOLD,
    SPAWN_BREADTH_THRESHOLD,
    SPAWN_PARALLELISM_THRESHOLD,
    get_llm_kwargs,
)
from core.blueprint import AgentBlueprint, SpawnBudget, compile_system_prompt
from core.cost_tracker import CostTracker
from tracing.event_logger import EventLogger
from tracing.event_types import AGENT_SPAWN_DECISION, AGENT_SPAWNED
from memory.blackboard import BlackboardClient
from memory.locks import SpawnLock
from memory.registry import AgentRegistry
from prompts.spawner import ASSESSMENT_PROMPT, DECOMPOSITION_PROMPT

logger = logging.getLogger("evolve.spawner")


@dataclass
class SpawnAssessment:
    """Result of an agent's self-assessment on whether to decompose."""

    breadth: float = 0.0
    depth: float = 0.0
    parallelism: float = 0.0
    reasoning: str = ""

    @property
    def should_spawn(self) -> bool:
        return (
            (self.breadth > SPAWN_BREADTH_THRESHOLD and self.parallelism > SPAWN_PARALLELISM_THRESHOLD)
            or self.breadth > SPAWN_AUTO_THRESHOLD
        )



class Spawner:
    """Handles spawn assessment, reuse-first lookup, budget locking, and agent creation."""

    def __init__(
        self,
        registry: AgentRegistry,
        event_logger: EventLogger,
        cost_tracker: CostTracker,
        spawn_lock: SpawnLock,
        session_id: str,
    ):
        self.registry = registry
        self.event_logger = event_logger
        self.cost_tracker = cost_tracker
        self.spawn_lock = spawn_lock
        self.session_id = session_id

    async def assess_task(self, task: str, model: str = "gpt-4o") -> SpawnAssessment:
        """Ask the LLM to score the task on breadth/depth/parallelism."""
        kwargs = get_llm_kwargs(model)
        kwargs["messages"] = [{"role": "user", "content": ASSESSMENT_PROMPT.format(task=task)}]
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 0.2
        response = await litellm.acompletion(**kwargs)
        usage = response.usage
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return SpawnAssessment(
            breadth=float(data.get("breadth", 0)),
            depth=float(data.get("depth", 0)),
            parallelism=float(data.get("parallelism", 0)),
            reasoning=data.get("reasoning", ""),
        ), usage

    async def plan_decomposition(self, task: str, model: str = "gpt-4o") -> tuple[list[dict], object]:
        """Ask the LLM to decompose a task into sub-agent specs."""
        kwargs = get_llm_kwargs(model)
        kwargs["messages"] = [{"role": "user", "content": DECOMPOSITION_PROMPT.format(task=task)}]
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 0.3
        response = await litellm.acompletion(**kwargs)
        usage = response.usage
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return data.get("agents", []), usage

    async def maybe_spawn(
        self,
        parent_blueprint: AgentBlueprint,
        task_spec: dict,
    ) -> AgentBlueprint | None:
        """Try to reuse an existing agent, resurrect a hibernated one, or spawn new."""

        required_skills = task_spec.get("expertise", [])

        # 1. Check for a living agent that already has the needed skills
        existing = await self.registry.find_agent(
            session_id=self.session_id,
            status="alive",
            capabilities=required_skills,
        )
        if existing and existing["_id"] != parent_blueprint.agent_id:
            # Reuse by dispatching a task to this agent
            logger.info("Reusing alive agent %s for task (parent=%s)", existing["name"], parent_blueprint.name)
            return None  # Caller should dispatch_task instead

        # 2. Check for a hibernated agent
        hibernated = await self.registry.find_agent(
            session_id=self.session_id,
            status="hibernated",
            capabilities=required_skills,
        )
        if hibernated:
            # Resurrect — caller handles the lifecycle.resurrect() call
            logger.info("Found hibernated agent %s for reuse (parent=%s)", hibernated["name"], parent_blueprint.name)
            return None

        # 3. Spawn new — with lock
        return await self._create_agent(parent_blueprint, task_spec)

    async def _create_agent(
        self,
        parent: AgentBlueprint,
        spec: dict,
    ) -> AgentBlueprint | None:
        """Create a new agent under a distributed lock."""
        async with self.spawn_lock.acquire(self.session_id) as acquired:
            if not acquired:
                logger.warning("Could not acquire spawn lock for session %s", self.session_id)
                return None

            # Check global cap
            alive_count = await self.registry.count_alive(self.session_id)
            if alive_count >= MAX_GLOBAL_AGENTS:
                logger.warning("Spawn denied: global cap reached (%d/%d)", alive_count, MAX_GLOBAL_AGENTS)
                await self.event_logger.log_event(
                    session_id=self.session_id,
                    agent_id=parent.agent_id,
                    agent_name=parent.name,
                    depth=parent.depth,
                    event_type=AGENT_SPAWN_DECISION,
                    payload={"decision": "denied", "reason": "global_cap_reached"},
                )
                return None

            # Check parent budget
            if not parent.spawn_budget.can_spawn:
                logger.warning("Spawn denied: parent %s budget exhausted", parent.name)
                await self.event_logger.log_event(
                    session_id=self.session_id,
                    agent_id=parent.agent_id,
                    agent_name=parent.name,
                    depth=parent.depth,
                    event_type=AGENT_SPAWN_DECISION,
                    payload={"decision": "denied", "reason": "budget_exhausted"},
                )
                return None

            # Build the child blueprint
            child_depth = parent.depth + 1
            tools_allowed = spec.get("tools_needed", [])
            can_spawn = spec.get("needs_spawn", False) and child_depth < MAX_DEPTH

            if can_spawn:
                tools_allowed.append("spawn_agent")

            child_budget = SpawnBudget(
                max_children=max(0, parent.spawn_budget.max_children // 2),
                max_depth_remaining=parent.spawn_budget.max_depth_remaining - 1,
                remaining_global=MAX_GLOBAL_AGENTS - alive_count - 1,
            )

            child = AgentBlueprint(
                name=spec.get("name", f"Agent-depth{child_depth}"),
                lineage=parent.lineage + [],
                role=spec.get("role", "engineer"),
                persona=spec.get("persona", "A focused specialist"),
                expertise=spec.get("expertise", []),
                verbosity=spec.get("verbosity", "normal"),
                tools_allowed=tools_allowed,
                tools_denied=[] if can_spawn else ["spawn_agent"],
                model=spec.get("model", ""),
                ttl_seconds=spec.get("ttl_seconds", parent.ttl_seconds),
                max_steps=spec.get("max_steps", parent.max_steps),
                priority=spec.get("priority", "normal"),
                spawn_budget=child_budget,
                task=spec.get("task", ""),
                success_criteria=spec.get("success_criteria", ""),
                output_format=spec.get("output_format", "markdown"),
                depth=child_depth,
            )
            child.lineage = parent.lineage + [child.agent_id]

            # Compile system prompt
            system_prompt = compile_system_prompt(child)

            # Register in MongoDB
            await self.registry.register(
                child, session_id=self.session_id, parent_id=parent.agent_id
            )
            await self.registry.update(child.agent_id, system_prompt_compiled=system_prompt)
            await self.registry.add_child(parent.agent_id, child.agent_id)

            # Decrement parent budget
            parent.spawn_budget.max_children -= 1
            await self.registry.update(
                parent.agent_id,
                spawn_budget_remaining=parent.spawn_budget.max_children,
            )

            # Log spawn event
            await self.event_logger.log_event(
                session_id=self.session_id,
                agent_id=child.agent_id,
                agent_name=child.name,
                depth=child.depth,
                event_type=AGENT_SPAWNED,
                payload={
                    "parent_id": parent.agent_id,
                    "role": child.role,
                    "task": child.task,
                    "model": child.model,
                    "spawn_reason": spec.get("spawn_reason", "decomposition"),
                },
            )

            return child
