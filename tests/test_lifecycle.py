"""Tests for agent lifecycle — TTL, hibernation, death, resurrection."""

import pytest

from core.lifecycle import AgentExpired, AgentLifecycle, resurrect


@pytest.mark.asyncio
class TestAgentLifecycle:
    async def test_start_sets_heartbeat(self, mock_redis, mock_registry, mock_event_logger):
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        await lc.start()
        mock_redis.setex.assert_called_once_with("heartbeat:agent-1", 60, "alive")

    async def test_check_alive_returns_true(self, mock_redis, mock_registry, mock_event_logger):
        mock_redis.exists.return_value = True
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        assert await lc.check_alive() is True

    async def test_step_raises_when_ttl_expired(self, mock_redis, mock_registry, mock_event_logger):
        mock_redis.exists.return_value = False
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        with pytest.raises(AgentExpired, match="TTL expired"):
            await lc.step()

    async def test_step_raises_when_max_steps_exceeded(self, mock_redis, mock_registry, mock_event_logger):
        mock_redis.exists.return_value = True
        mock_registry.increment_steps.return_value = 10  # equals max_steps
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        with pytest.raises(AgentExpired, match="max steps"):
            await lc.step()

    async def test_step_succeeds_within_limits(self, mock_redis, mock_registry, mock_event_logger):
        mock_redis.exists.return_value = True
        mock_registry.increment_steps.return_value = 5
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        steps = await lc.step()
        assert steps == 5

    async def test_hibernate_updates_status(self, mock_redis, mock_registry, mock_event_logger):
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        await lc.hibernate(reason="ttl_expired")
        mock_redis.delete.assert_called_with("heartbeat:agent-1")
        mock_registry.update.assert_called_once()
        call_kwargs = mock_registry.update.call_args
        assert call_kwargs[1]["status"] == "hibernated"

    async def test_complete_updates_status(self, mock_redis, mock_registry, mock_event_logger):
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        await lc.complete("Final output text")
        mock_registry.update.assert_called_once()
        call_kwargs = mock_registry.update.call_args
        assert call_kwargs[1]["status"] == "completed"
        assert call_kwargs[1]["final_output"] == "Final output text"

    async def test_die_updates_status(self, mock_redis, mock_registry, mock_event_logger):
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        await lc.die(reason="budget_exhausted")
        mock_registry.update.assert_called_once()
        call_kwargs = mock_registry.update.call_args
        assert call_kwargs[1]["status"] == "dead"
        assert call_kwargs[1]["exit_reason"] == "budget_exhausted"

    async def test_extend_ttl(self, mock_redis, mock_registry, mock_event_logger):
        mock_redis.ttl.return_value = 100
        lc = AgentLifecycle(
            agent_id="agent-1", agent_name="TestAgent", depth=1,
            ttl_seconds=60, max_steps=10, session_id="session-1",
            registry=mock_registry, event_logger=mock_event_logger, redis=mock_redis,
        )
        await lc.extend_ttl(60)
        mock_redis.expire.assert_called_once_with("heartbeat:agent-1", 160)


@pytest.mark.asyncio
class TestResurrect:
    async def test_resurrect_hibernated_agent(self, mock_registry, mock_event_logger, mock_redis):
        mock_registry.find_by_id.return_value = {
            "status": "hibernated",
            "blueprint": {
                "agent_id": "agent-1",
                "name": "SleepingAgent",
                "role": "researcher",
                "persona": "A researcher",
                "expertise": ["ml"],
                "tools_allowed": ["web_search"],
                "tools_denied": [],
                "model": "azure/gpt-4o",
                "ttl_seconds": 300,
                "max_steps": 20,
                "task": "Research models",
                "success_criteria": "Found models",
                "output_format": "markdown",
                "depth": 1,
                "verbosity": "normal",
                "risk_tolerance": "moderate",
                "priority": "normal",
            },
            "saved_state": {"partial_results": "Found 2 models", "last_step": "step 3"},
        }

        bp, prompt, state = await resurrect(
            agent_id="agent-1",
            registry=mock_registry,
            event_logger=mock_event_logger,
            redis=mock_redis,
            session_id="session-1",
        )
        assert bp.name == "SleepingAgent"
        assert "RESURRECTION CONTEXT" in prompt
        assert "Found 2 models" in prompt
        mock_redis.setex.assert_called_once()

    async def test_resurrect_non_hibernated_raises(self, mock_registry, mock_event_logger, mock_redis):
        mock_registry.find_by_id.return_value = {"status": "alive"}
        with pytest.raises(ValueError, match="not hibernated"):
            await resurrect(
                agent_id="agent-1",
                registry=mock_registry,
                event_logger=mock_event_logger,
                redis=mock_redis,
                session_id="session-1",
            )
