"""AgentBlueprint schema and system prompt compiler."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TTL_SECONDS,
    ROLE_MODEL_DEFAULTS,
)
from prompts.agent import (
    AGENT_SYSTEM_PROMPT_TEMPLATE,
    SPAWN_ALLOWED_SECTION,
    SPAWN_DENIED_SECTION,
)


@dataclass
class MemoryScope:
    """Defines what namespaces an agent can read/write on the blackboard."""

    write_ns: str  # always "agent:{self.id}"
    read_ns: list[str] = field(default_factory=lambda: ["shared"])
    share_policy: str = "auto_conclusions"  # "manual" | "auto_conclusions" | "full_transparency"


@dataclass
class SpawnBudget:
    """Inherited budget for how many children an agent can create."""

    max_children: int = 0
    max_depth_remaining: int = 0
    remaining_global: int = 0  # snapshot at spawn time

    @property
    def can_spawn(self) -> bool:
        return (
            self.max_children > 0
            and self.max_depth_remaining > 0
            and self.remaining_global > 0
        )


@dataclass
class AgentBlueprint:
    """The DNA of an agent — everything needed to create or resurrect it."""

    # Identity
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    lineage: list[str] = field(default_factory=list)

    # Persona & Behavior
    role: str = "engineer"
    persona: str = "A capable and focused AI agent"
    expertise: list[str] = field(default_factory=list)
    verbosity: str = "normal"  # "minimal" | "normal" | "detailed"
    risk_tolerance: str = "moderate"  # "conservative" | "moderate" | "aggressive"

    # Capabilities
    tools_allowed: list[str] = field(default_factory=list)
    tools_denied: list[str] = field(default_factory=list)
    model: str = ""  # empty = auto-select from role

    # Memory
    memory_scope: MemoryScope = field(default_factory=lambda: MemoryScope(write_ns=""))

    # Lifecycle
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_steps: int = DEFAULT_MAX_STEPS
    priority: str = "normal"  # "critical" | "normal" | "background"
    spawn_budget: SpawnBudget = field(default_factory=SpawnBudget)

    # Task
    task: str = ""
    success_criteria: str = ""
    output_format: str = "markdown"  # "json" | "markdown" | "code" | "structured_report"

    # Depth in the agent tree (0 = genesis)
    depth: int = 0

    def __post_init__(self):
        if not self.model:
            self.model = ROLE_MODEL_DEFAULTS.get(self.role, "gpt-4o")
        if not self.memory_scope.write_ns:
            self.memory_scope.write_ns = f"agent:{self.agent_id}"
        if not self.lineage:
            self.lineage = [self.agent_id]


def compile_system_prompt(blueprint: AgentBlueprint) -> str:
    """Convert an AgentBlueprint into a system prompt string."""

    if "spawn_agent" in blueprint.tools_allowed:
        spawn_section = SPAWN_ALLOWED_SECTION.format(
            max_children=blueprint.spawn_budget.max_children,
            max_depth_remaining=blueprint.spawn_budget.max_depth_remaining,
        )
    else:
        spawn_section = SPAWN_DENIED_SECTION

    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        name=blueprint.name,
        role=blueprint.role,
        persona=blueprint.persona,
        expertise=', '.join(blueprint.expertise) or 'general',
        verbosity=blueprint.verbosity,
        task=blueprint.task,
        success_criteria=blueprint.success_criteria,
        output_format=blueprint.output_format,
        max_steps=blueprint.max_steps,
        tools_allowed=', '.join(blueprint.tools_allowed) or 'none',
        tools_denied=', '.join(blueprint.tools_denied) or 'none',
        risk_tolerance=blueprint.risk_tolerance,
        spawn_section=spawn_section,
        write_ns=blueprint.memory_scope.write_ns,
        read_ns=', '.join(blueprint.memory_scope.read_ns),
        share_policy=blueprint.memory_scope.share_policy,
    )
