"""Integration test for parallel vs sequential spawning.

Tests the tool_node logic directly with mocked spawn calls to verify:
1. Parallel spawns run concurrently (via asyncio.gather)
2. Sequential spawns (parallel=false) run one after another
3. Mixed mode: sequential first, then parallel

Run with:
    python -m pytest tests/test_spawn_modes.py -v -s
"""

import asyncio
import json
import time
import pytest

from langchain_core.messages import AIMessage, ToolMessage

# We need to patch _handle_spawn to track timing instead of actually running agent graphs


@pytest.fixture
def spawn_timing_tracker():
    """Returns a list that collects (agent_name, start_time, end_time) tuples."""
    return []


@pytest.fixture
def make_tool_calls():
    """Helper to create AIMessage with spawn_agent tool calls."""
    def _make(specs: list[dict]) -> AIMessage:
        tool_calls = []
        for i, spec in enumerate(specs):
            tool_calls.append({
                "id": f"call_{i}",
                "name": "spawn_agent",
                "args": spec,
            })
        return AIMessage(content="Spawning agents", tool_calls=tool_calls)
    return _make


@pytest.mark.asyncio
async def test_parallel_spawns_run_concurrently(make_tool_calls, spawn_timing_tracker):
    """When parallel=true (default), multiple spawns should overlap in time."""
    import core.agent as agent_module

    DELAY = 0.3  # seconds each spawn "takes"

    original_handle_spawn = agent_module._handle_spawn

    async def mock_handle_spawn(args, parent_bp, spawner, deps):
        name = args.get("name", "unknown")
        start = time.monotonic()
        await asyncio.sleep(DELAY)  # Simulate work
        end = time.monotonic()
        spawn_timing_tracker.append((name, start, end))
        return {"tool": "spawn_agent", "status": "completed", "agent_name": name, "output": f"Done: {name}"}

    agent_module._handle_spawn = mock_handle_spawn
    try:
        # Create 3 parallel spawn calls (parallel=true is default)
        ai_msg = make_tool_calls([
            {"name": "AgentA", "role": "researcher", "task": "task A", "success_criteria": "done"},
            {"name": "AgentB", "role": "engineer", "task": "task B", "success_criteria": "done"},
            {"name": "AgentC", "role": "critic", "task": "task C", "success_criteria": "done"},
        ])

        # Build minimal state and config
        state = _build_state(ai_msg)
        config = _build_config()

        wall_start = time.monotonic()
        result = await agent_module.tool_node(state, config)
        wall_end = time.monotonic()
        wall_time = wall_end - wall_start

        # Verify all 3 completed
        assert len(result["messages"]) == 3
        assert len(spawn_timing_tracker) == 3

        # Wall time should be ~DELAY (parallel), not ~3*DELAY (sequential)
        # Allow some slack for overhead
        assert wall_time < DELAY * 2, (
            f"Parallel spawns took {wall_time:.2f}s — expected ~{DELAY}s, got sequential behavior"
        )
        print(f"\n✓ Parallel: 3 spawns in {wall_time:.2f}s (each takes {DELAY}s)")
        for name, s, e in spawn_timing_tracker:
            print(f"  {name}: {s - wall_start:.3f}s → {e - wall_start:.3f}s")

    finally:
        agent_module._handle_spawn = original_handle_spawn


@pytest.mark.asyncio
async def test_sequential_spawns_run_in_order(make_tool_calls, spawn_timing_tracker):
    """When parallel=false, spawns should run one after another."""
    import core.agent as agent_module

    DELAY = 0.2

    async def mock_handle_spawn(args, parent_bp, spawner, deps):
        name = args.get("name", "unknown")
        start = time.monotonic()
        await asyncio.sleep(DELAY)
        end = time.monotonic()
        spawn_timing_tracker.append((name, start, end))
        return {"tool": "spawn_agent", "status": "completed", "agent_name": name, "output": f"Done: {name}"}

    original = agent_module._handle_spawn
    agent_module._handle_spawn = mock_handle_spawn
    try:
        ai_msg = make_tool_calls([
            {"name": "SeqA", "role": "researcher", "task": "task A", "success_criteria": "done", "parallel": False},
            {"name": "SeqB", "role": "engineer", "task": "task B", "success_criteria": "done", "parallel": False},
            {"name": "SeqC", "role": "critic", "task": "task C", "success_criteria": "done", "parallel": False},
        ])

        state = _build_state(ai_msg)
        config = _build_config()

        wall_start = time.monotonic()
        result = await agent_module.tool_node(state, config)
        wall_end = time.monotonic()
        wall_time = wall_end - wall_start

        assert len(result["messages"]) == 3
        assert len(spawn_timing_tracker) == 3

        # Wall time should be ~3*DELAY (sequential)
        assert wall_time >= DELAY * 2.5, (
            f"Sequential spawns took {wall_time:.2f}s — expected ~{DELAY * 3}s, ran in parallel?"
        )

        # Verify ordering: each starts after the previous ends
        for i in range(1, len(spawn_timing_tracker)):
            prev_end = spawn_timing_tracker[i - 1][2]
            curr_start = spawn_timing_tracker[i][1]
            assert curr_start >= prev_end - 0.01, (
                f"{spawn_timing_tracker[i][0]} started before {spawn_timing_tracker[i-1][0]} finished"
            )

        print(f"\n✓ Sequential: 3 spawns in {wall_time:.2f}s (each takes {DELAY}s)")
        for name, s, e in spawn_timing_tracker:
            print(f"  {name}: {s - wall_start:.3f}s → {e - wall_start:.3f}s")

    finally:
        agent_module._handle_spawn = original


@pytest.mark.asyncio
async def test_mixed_mode_sequential_then_parallel(make_tool_calls, spawn_timing_tracker):
    """Sequential spawns run first (in order), then parallel spawns fan out."""
    import core.agent as agent_module

    DELAY = 0.2

    async def mock_handle_spawn(args, parent_bp, spawner, deps):
        name = args.get("name", "unknown")
        start = time.monotonic()
        await asyncio.sleep(DELAY)
        end = time.monotonic()
        spawn_timing_tracker.append((name, start, end))
        return {"tool": "spawn_agent", "status": "completed", "agent_name": name, "output": f"Done: {name}"}

    original = agent_module._handle_spawn
    agent_module._handle_spawn = mock_handle_spawn
    try:
        ai_msg = make_tool_calls([
            {"name": "SeqFirst", "role": "researcher", "task": "must run first", "success_criteria": "done", "parallel": False},
            {"name": "ParA", "role": "engineer", "task": "parallel A", "success_criteria": "done", "parallel": True},
            {"name": "ParB", "role": "critic", "task": "parallel B", "success_criteria": "done", "parallel": True},
        ])

        state = _build_state(ai_msg)
        config = _build_config()

        wall_start = time.monotonic()
        result = await agent_module.tool_node(state, config)
        wall_end = time.monotonic()
        wall_time = wall_end - wall_start

        assert len(result["messages"]) == 3
        assert len(spawn_timing_tracker) == 3

        # SeqFirst should complete before ParA and ParB start
        seq_entry = next(e for e in spawn_timing_tracker if e[0] == "SeqFirst")
        par_entries = [e for e in spawn_timing_tracker if e[0].startswith("Par")]

        for par_name, par_start, _ in par_entries:
            assert par_start >= seq_entry[2] - 0.01, (
                f"{par_name} started at {par_start:.3f} before SeqFirst ended at {seq_entry[2]:.3f}"
            )

        # ParA and ParB should overlap (start at similar times)
        if len(par_entries) == 2:
            start_diff = abs(par_entries[0][1] - par_entries[1][1])
            assert start_diff < DELAY * 0.5, (
                f"Parallel agents started {start_diff:.3f}s apart — not parallel"
            )

        # Total: ~DELAY (seq) + ~DELAY (parallel pair) = ~2*DELAY
        assert wall_time < DELAY * 3, (
            f"Mixed mode took {wall_time:.2f}s — expected ~{DELAY * 2}s"
        )

        print(f"\n✓ Mixed: 1 sequential + 2 parallel in {wall_time:.2f}s (each takes {DELAY}s)")
        for name, s, e in spawn_timing_tracker:
            print(f"  {name}: {s - wall_start:.3f}s → {e - wall_start:.3f}s")

    finally:
        agent_module._handle_spawn = original


# ── Helpers ─────────────────────────────────────────────────────

def _build_state(last_msg: AIMessage) -> dict:
    """Build a minimal AgentState for tool_node."""
    return {
        "messages": [last_msg],
        "agent_id": "test-agent",
        "agent_name": "TestAgent",
        "session_id": "test-session",
        "depth": 0,
        "model": "ollama/qwen3:8b",
        "tools_allowed": ["spawn_agent"],
        "tools_denied": [],
        "final_output": "",
        "step_count": 0,
        "max_steps": 30,
        "status": "running",
    }


def _build_config() -> dict:
    """Build a minimal RunnableConfig with stub deps."""
    from core.blueprint import AgentBlueprint, SpawnBudget
    from core.spawner import Spawner
    from core.cost_tracker import CostTracker

    cost_tracker = CostTracker(max_tokens=100000)

    blueprint = AgentBlueprint(
        name="TestAgent", role="orchestrator", task="test",
        success_criteria="done", tools_allowed=["spawn_agent"],
        spawn_budget=SpawnBudget(max_children=8, max_depth_remaining=4, remaining_global=20),
    )

    spawner = Spawner(session_id="test-session")

    return {
        "configurable": {
            "thread_id": "test-agent",
            "cost_tracker": cost_tracker,
            "blueprint": blueprint,
            "spawner": spawner,
            "session_id": "test-session",
            "checkpointer": None,
        }
    }
