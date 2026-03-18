"""MemoryScope helpers and namespace logic."""

from __future__ import annotations

from core.blueprint import MemoryScope


def build_scope_for_agent(
    agent_id: str,
    team_tags: list[str] | None = None,
    share_policy: str = "auto_conclusions",
) -> MemoryScope:
    """Build a MemoryScope for a newly created agent."""
    read_ns = ["shared", f"agent:{agent_id}"]
    if team_tags:
        read_ns.extend(f"team:{tag}" for tag in team_tags)

    return MemoryScope(
        write_ns=f"agent:{agent_id}",
        read_ns=read_ns,
        share_policy=share_policy,
    )
