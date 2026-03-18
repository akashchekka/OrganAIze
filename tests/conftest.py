"""Shared test fixtures for the Evolve test suite."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.blueprint import AgentBlueprint, MemoryScope, SpawnBudget


@pytest.fixture
def sample_blueprint() -> AgentBlueprint:
    """A basic agent blueprint for testing."""
    return AgentBlueprint(
        agent_id="test-agent-001",
        name="TestAgent",
        role="engineer",
        persona="A test agent",
        expertise=["testing", "python"],
        tools_allowed=["web_search", "code_execute"],
        tools_denied=["file_delete"],
        model="azure/gpt-4o",
        memory_scope=MemoryScope(
            write_ns="agent:test-agent-001",
            read_ns=["shared", "agent:test-agent-001"],
            share_policy="auto_conclusions",
        ),
        ttl_seconds=60,
        max_steps=10,
        spawn_budget=SpawnBudget(max_children=4, max_depth_remaining=3, remaining_global=18),
        task="Write unit tests",
        success_criteria="All tests pass",
        output_format="code",
        depth=1,
    )


@pytest.fixture
def genesis_blueprint() -> AgentBlueprint:
    """A genesis-level blueprint with full spawn budget."""
    return AgentBlueprint(
        agent_id="genesis-001",
        name="Genesis-Orchestrator",
        role="orchestrator",
        persona="Master orchestrator",
        expertise=["decomposition", "orchestration"],
        tools_allowed=["spawn_agent", "web_search", "share_finding", "read_shared"],
        tools_denied=[],
        model="azure/gpt-4o",
        ttl_seconds=600,
        max_steps=60,
        spawn_budget=SpawnBudget(max_children=8, max_depth_remaining=4, remaining_global=20),
        task="Build an app",
        success_criteria="App is functional",
        output_format="markdown",
        depth=0,
    )


@pytest.fixture
def leaf_blueprint() -> AgentBlueprint:
    """A depth-4 agent that cannot spawn."""
    return AgentBlueprint(
        agent_id="leaf-001",
        name="LeafAgent",
        role="engineer",
        persona="A leaf worker",
        expertise=["coding"],
        tools_allowed=["code_execute"],
        tools_denied=["spawn_agent"],
        model="azure/gpt-4o-mini",
        spawn_budget=SpawnBudget(max_children=0, max_depth_remaining=0, remaining_global=5),
        task="Write a function",
        success_criteria="Function works",
        depth=4,
    )


@pytest.fixture
def mock_registry():
    """Mock AgentRegistry for tests that don't need MongoDB."""
    registry = AsyncMock()
    registry.register = AsyncMock(return_value={"_id": "test-agent-001"})
    registry.find_by_id = AsyncMock(return_value=None)
    registry.update = AsyncMock()
    registry.add_child = AsyncMock()
    registry.increment_steps = AsyncMock(return_value=1)
    registry.count_alive = AsyncMock(return_value=3)
    registry.find_agent = AsyncMock(return_value=None)
    registry.get_session_agents = AsyncMock(return_value=[])
    return registry


@pytest.fixture
def mock_event_logger():
    """Mock EventLogger for tests that don't need MongoDB."""
    logger = AsyncMock()
    logger.log = AsyncMock()
    logger.log_event = AsyncMock()
    logger.ensure_indexes = AsyncMock()
    return logger


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    redis = AsyncMock()
    redis.setex = AsyncMock()
    redis.exists = AsyncMock(return_value=True)
    redis.delete = AsyncMock()
    redis.ttl = AsyncMock(return_value=200)
    redis.expire = AsyncMock()
    redis.set = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.publish = AsyncMock()
    redis.keys = AsyncMock(return_value=[])
    redis.scan_iter = MagicMock(return_value=iter([]))
    redis.lock = MagicMock()
    return redis
