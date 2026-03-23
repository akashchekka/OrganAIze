"""Shared test fixtures for the OrganAIze test suite."""

import pytest

from core.blueprint import AgentBlueprint, SpawnBudget


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
        tools_allowed=["spawn_agent", "web_search"],
        tools_denied=[],
        model="azure/gpt-4o",
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
