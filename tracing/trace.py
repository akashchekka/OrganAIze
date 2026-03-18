"""Causal chain reconstruction from the event log."""

from __future__ import annotations

from tracing.event_logger import EventLogger


async def build_spawn_tree(event_logger: EventLogger, session_id: str) -> dict:
    """Reconstruct the agent hierarchy as a nested tree from spawn events.

    Returns:
        {
            "agent_id": "genesis_001",
            "name": "Genesis-Orchestrator",
            "children": [
                {"agent_id": "...", "name": "...", "children": [...]},
                ...
            ]
        }
    """
    spawns = await event_logger.get_spawn_tree(session_id)

    nodes: dict[str, dict] = {}
    root_id = None

    for event in spawns:
        agent_id = event["agent_id"]
        parent_id = event["payload"].get("parent_id")
        node = {
            "agent_id": agent_id,
            "name": event["agent_name"],
            "role": event["payload"].get("role", ""),
            "depth": event["depth"],
            "children": [],
        }
        nodes[agent_id] = node

        if parent_id is None:
            root_id = agent_id
        elif parent_id in nodes:
            nodes[parent_id]["children"].append(node)

    return nodes.get(root_id, {}) if root_id else {}


async def get_critical_path(event_logger: EventLogger, session_id: str) -> list[dict]:
    """Trace the critical path from final output back through causal chain."""
    events = await event_logger.get_session_events(session_id)

    # Find output events
    outputs = [e for e in events if e["event_type"] == "agent.output"]
    if not outputs:
        return []

    # Trace back from the last output (genesis) through parent_event_ids
    path = []
    current = outputs[-1]
    event_map = {e["event_id"]: e for e in events}

    while current:
        path.append({
            "event_id": current["event_id"],
            "agent_name": current["agent_name"],
            "event_type": current["event_type"],
            "timestamp": current["timestamp"],
        })
        parent_id = current.get("parent_event_id")
        current = event_map.get(parent_id) if parent_id else None

    path.reverse()
    return path
