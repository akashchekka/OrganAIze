"""Tests for spawner — budget enforcement and spawn creation."""

import pytest

from core.blueprint import AgentBlueprint, SpawnBudget
from core.spawner import Spawner


class TestSpawnerSpawn:
    def _make_spawner(self):
        return Spawner(session_id="session-1")

    def test_spawn_new_agent(self, genesis_blueprint):
        spawner = self._make_spawner()

        spec = {
            "name": "NewAgent",
            "role": "researcher",
            "persona": "A researcher",
            "expertise": ["ml"],
            "task": "Research models",
            "success_criteria": "Found models",
            "output_format": "markdown",
            "tools_needed": ["web_search"],
            "needs_spawn": False,
        }

        result = spawner.spawn(genesis_blueprint, spec)
        assert result is not None
        assert result.name == "NewAgent"
        assert result.role == "researcher"
        assert result.depth == 1

    def test_spawn_denied_global_cap(self, genesis_blueprint):
        spawner = self._make_spawner()
        from config import MAX_GLOBAL_AGENTS
        spawner.agents_created = MAX_GLOBAL_AGENTS  # At cap

        result = spawner.spawn(genesis_blueprint, {"task": "test"})
        assert result is None

    def test_spawn_denied_budget_exhausted(self):
        bp = AgentBlueprint(
            agent_id="parent-1",
            name="ExhaustedParent",
            spawn_budget=SpawnBudget(max_children=0, max_depth_remaining=3, remaining_global=10),
            depth=1,
        )
        spawner = self._make_spawner()

        result = spawner.spawn(bp, {"task": "test"})
        assert result is None

    def test_child_budget_halves(self, genesis_blueprint):
        spawner = self._make_spawner()

        spec = {
            "name": "Child",
            "role": "engineer",
            "task": "Build something",
            "success_criteria": "Done",
            "needs_spawn": True,
        }

        child = spawner.spawn(genesis_blueprint, spec)
        assert child is not None
        # Genesis has max_children=8, child should get 8//2=4
        assert child.spawn_budget.max_children == 4
        assert child.spawn_budget.max_depth_remaining == 3  # 4-1

    def test_child_depth_increments(self, genesis_blueprint):
        spawner = self._make_spawner()

        child = spawner.spawn(genesis_blueprint, {
            "name": "Child", "role": "engineer", "task": "test", "success_criteria": "done",
        })
        assert child.depth == 1  # genesis is depth 0

    def test_spawn_agent_denied_for_deep_child(self):
        parent = AgentBlueprint(
            agent_id="deep-parent",
            name="DeepParent",
            depth=3,
            spawn_budget=SpawnBudget(max_children=2, max_depth_remaining=1, remaining_global=10),
        )
        spawner = self._make_spawner()

        child = spawner.spawn(parent, {
            "name": "LeafChild", "role": "engineer",
            "task": "test", "success_criteria": "done",
            "needs_spawn": True,  # Wants to spawn but at MAX_DEPTH
        })
        assert child is not None
        # Child is at depth 4, which equals MAX_DEPTH, so spawn_agent should be denied
        assert "spawn_agent" in child.tools_denied

    def test_agents_created_counter_increments(self, genesis_blueprint):
        spawner = self._make_spawner()
        assert spawner.agents_created == 0

        spawner.spawn(genesis_blueprint, {
            "name": "A", "task": "t1", "success_criteria": "done",
        })
        assert spawner.agents_created == 1

        spawner.spawn(genesis_blueprint, {
            "name": "B", "task": "t2", "success_criteria": "done",
        })
        assert spawner.agents_created == 2
