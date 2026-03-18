"""Tests for spawner — budget enforcement, reuse-first, spawn creation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from core.blueprint import AgentBlueprint, SpawnBudget
from core.spawner import SpawnAssessment, Spawner


class TestSpawnAssessment:
    def test_should_spawn_high_breadth_and_parallelism(self):
        a = SpawnAssessment(breadth=0.8, depth=0.5, parallelism=0.7)
        assert a.should_spawn is True

    def test_should_not_spawn_low_breadth(self):
        a = SpawnAssessment(breadth=0.3, depth=0.9, parallelism=0.9)
        assert a.should_spawn is False

    def test_should_spawn_very_high_breadth(self):
        a = SpawnAssessment(breadth=0.9, depth=0.2, parallelism=0.1)
        assert a.should_spawn is True

    def test_should_not_spawn_moderate_breadth_low_parallelism(self):
        a = SpawnAssessment(breadth=0.7, depth=0.5, parallelism=0.2)
        assert a.should_spawn is False

    def test_threshold_boundary(self):
        # Exactly at threshold: breadth=0.6, parallelism=0.4
        a = SpawnAssessment(breadth=0.6, depth=0.5, parallelism=0.4)
        assert a.should_spawn is False  # > not >=


@pytest.mark.asyncio
class TestSpawnerMaybeSpawn:
    def _make_spawner(self, mock_registry, mock_event_logger):
        mock_lock = MagicMock()

        @asynccontextmanager
        async def fake_acquire(session_id):
            yield True

        mock_lock.acquire = fake_acquire

        return Spawner(
            registry=mock_registry,
            event_logger=mock_event_logger,
            cost_tracker=MagicMock(),
            spawn_lock=mock_lock,
            session_id="session-1",
        )

    async def test_reuse_alive_agent(self, mock_registry, mock_event_logger, genesis_blueprint):
        mock_registry.find_agent.return_value = {"_id": "existing-alive", "name": "AliveAgent"}
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        result = await spawner.maybe_spawn(genesis_blueprint, {"expertise": ["ml"]})
        assert result is None  # Should reuse, not spawn

    async def test_reuse_hibernated_agent(self, mock_registry, mock_event_logger, genesis_blueprint):
        # First call (alive) returns None, second call (hibernated) returns a match
        mock_registry.find_agent.side_effect = [None, {"_id": "hibernated-1", "name": "SleepyAgent"}]
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        result = await spawner.maybe_spawn(genesis_blueprint, {"expertise": ["ml"]})
        assert result is None  # Should resurrect, not spawn

    async def test_spawn_new_when_no_existing(self, mock_registry, mock_event_logger, genesis_blueprint):
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 3
        spawner = self._make_spawner(mock_registry, mock_event_logger)

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

        result = await spawner.maybe_spawn(genesis_blueprint, spec)
        assert result is not None
        assert result.name == "NewAgent"
        assert result.role == "researcher"
        assert result.depth == 1

    async def test_spawn_denied_global_cap(self, mock_registry, mock_event_logger, genesis_blueprint):
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 20  # At cap
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        result = await spawner.maybe_spawn(genesis_blueprint, {"task": "test"})
        assert result is None

    async def test_spawn_denied_budget_exhausted(self, mock_registry, mock_event_logger):
        bp = AgentBlueprint(
            agent_id="parent-1",
            name="ExhaustedParent",
            spawn_budget=SpawnBudget(max_children=0, max_depth_remaining=3, remaining_global=10),
            depth=1,
        )
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 5
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        result = await spawner.maybe_spawn(bp, {"task": "test"})
        assert result is None

    async def test_child_budget_halves(self, mock_registry, mock_event_logger, genesis_blueprint):
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 3
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        spec = {
            "name": "Child",
            "role": "engineer",
            "task": "Build something",
            "success_criteria": "Done",
            "needs_spawn": True,
        }

        child = await spawner.maybe_spawn(genesis_blueprint, spec)
        assert child is not None
        # Genesis has max_children=8, child should get 8//2=4
        assert child.spawn_budget.max_children == 4
        assert child.spawn_budget.max_depth_remaining == 3  # 4-1

    async def test_child_depth_increments(self, mock_registry, mock_event_logger, genesis_blueprint):
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 3
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        child = await spawner.maybe_spawn(genesis_blueprint, {
            "name": "Child", "role": "engineer", "task": "test", "success_criteria": "done",
        })
        assert child.depth == 1  # genesis is depth 0

    async def test_spawn_agent_denied_for_deep_child(self, mock_registry, mock_event_logger):
        parent = AgentBlueprint(
            agent_id="deep-parent",
            name="DeepParent",
            depth=3,
            spawn_budget=SpawnBudget(max_children=2, max_depth_remaining=1, remaining_global=10),
        )
        mock_registry.find_agent.return_value = None
        mock_registry.count_alive.return_value = 5
        spawner = self._make_spawner(mock_registry, mock_event_logger)

        child = await spawner.maybe_spawn(parent, {
            "name": "LeafChild", "role": "engineer",
            "task": "test", "success_criteria": "done",
            "needs_spawn": True,  # Wants to spawn but at MAX_DEPTH
        })
        assert child is not None
        # Child is at depth 4, which equals MAX_DEPTH, so spawn_agent should be denied
        assert "spawn_agent" in child.tools_denied
