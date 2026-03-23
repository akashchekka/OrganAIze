"""Spawner — in-memory budget enforcement and agent creation."""

from __future__ import annotations

import logging

from config import (
    MAX_DEPTH,
    MAX_GLOBAL_AGENTS,
)
from core.blueprint import AgentBlueprint, SpawnBudget, compile_system_prompt

logger = logging.getLogger("organaize.spawner")


class Spawner:
    """Creates child agents with budget enforcement. No DB, no locks — single-process."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agents_created: int = 0

    def spawn(
        self,
        parent: AgentBlueprint,
        spec: dict,
    ) -> AgentBlueprint | None:
        """Create a new agent if budget allows. Pure in-memory."""

        # Check global cap
        if self.agents_created >= MAX_GLOBAL_AGENTS:
            logger.warning("Spawn denied: global cap reached (%d/%d)", self.agents_created, MAX_GLOBAL_AGENTS)
            return None

        # Check parent budget
        if not parent.spawn_budget.can_spawn:
            logger.warning("Spawn denied: parent %s budget exhausted", parent.name)
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
            remaining_global=MAX_GLOBAL_AGENTS - self.agents_created - 1,
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
            max_steps=spec.get("max_steps", parent.max_steps),
            spawn_budget=child_budget,
            task=spec.get("task", ""),
            success_criteria=spec.get("success_criteria", ""),
            output_format=spec.get("output_format", "markdown"),
            depth=child_depth,
        )
        child.lineage = parent.lineage + [child.agent_id]

        # Decrement parent budget
        parent.spawn_budget.max_children -= 1
        self.agents_created += 1

        logger.info("Spawned %s (role=%s, depth=%d) from %s",
                     child.name, child.role, child.depth, parent.name)
        return child
