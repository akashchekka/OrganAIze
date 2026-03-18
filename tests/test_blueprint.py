"""Tests for AgentBlueprint schema and system prompt compiler."""

import pytest

from core.blueprint import (
    AgentBlueprint,
    MemoryScope,
    SpawnBudget,
    compile_system_prompt,
)


class TestSpawnBudget:
    def test_can_spawn_all_positive(self):
        budget = SpawnBudget(max_children=4, max_depth_remaining=3, remaining_global=10)
        assert budget.can_spawn is True

    def test_cannot_spawn_no_children(self):
        budget = SpawnBudget(max_children=0, max_depth_remaining=3, remaining_global=10)
        assert budget.can_spawn is False

    def test_cannot_spawn_no_depth(self):
        budget = SpawnBudget(max_children=4, max_depth_remaining=0, remaining_global=10)
        assert budget.can_spawn is False

    def test_cannot_spawn_no_global(self):
        budget = SpawnBudget(max_children=4, max_depth_remaining=3, remaining_global=0)
        assert budget.can_spawn is False

    def test_cannot_spawn_all_zero(self):
        budget = SpawnBudget()
        assert budget.can_spawn is False


class TestAgentBlueprint:
    def test_default_values(self):
        bp = AgentBlueprint()
        assert bp.role == "engineer"
        assert bp.depth == 0
        assert bp.agent_id  # UUID generated
        assert bp.lineage == [bp.agent_id]

    def test_auto_model_selection(self):
        bp = AgentBlueprint(role="orchestrator")
        assert bp.model == "azure/gpt-4o"

        bp_qa = AgentBlueprint(role="qa")
        assert bp_qa.model == "azure/gpt-4o-mini"

    def test_explicit_model_overrides_default(self):
        bp = AgentBlueprint(role="qa", model="gpt-4o")
        assert bp.model == "gpt-4o"

    def test_memory_scope_auto_set(self):
        bp = AgentBlueprint(agent_id="abc-123")
        assert bp.memory_scope.write_ns == "agent:abc-123"

    def test_lineage_auto_set(self):
        bp = AgentBlueprint(agent_id="agent-x")
        assert bp.lineage == ["agent-x"]

    def test_lineage_preserved_if_set(self):
        bp = AgentBlueprint(agent_id="child", lineage=["genesis", "parent", "child"])
        assert bp.lineage == ["genesis", "parent", "child"]


class TestCompileSystemPrompt:
    def test_contains_agent_name(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "TestAgent" in prompt

    def test_contains_task(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "Write unit tests" in prompt

    def test_contains_success_criteria(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "All tests pass" in prompt

    def test_spawn_denied_for_non_spawning_agent(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "CANNOT spawn" in prompt

    def test_spawn_allowed_with_budget(self, genesis_blueprint):
        prompt = compile_system_prompt(genesis_blueprint)
        assert "CAN spawn" in prompt
        assert "8 children" in prompt

    def test_contains_tools(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "web_search" in prompt
        assert "code_execute" in prompt

    def test_contains_denied_tools(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "file_delete" in prompt

    def test_contains_communication_section(self, sample_blueprint):
        prompt = compile_system_prompt(sample_blueprint)
        assert "agent:test-agent-001" in prompt
        assert "auto_conclusions" in prompt
