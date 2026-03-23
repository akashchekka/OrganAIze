"""Tests for tool registry — access control and tool descriptions."""

import pytest

from tools.tool_registry import (
    ToolDenied,
    execute_tool,
    get_tool_descriptions,
    validate_tool_access,
)


class TestValidateToolAccess:
    def test_allowed_tool_passes(self):
        # Should not raise
        validate_tool_access(
            tool_name="web_search",
            agent_tools_allowed=["web_search", "code_execute"],
            agent_tools_denied=[],
            agent_depth=1,
        )

    def test_denied_tool_raises(self):
        with pytest.raises(ToolDenied, match="explicitly denied"):
            validate_tool_access(
                tool_name="file_delete",
                agent_tools_allowed=["file_delete"],  # allowed but also denied
                agent_tools_denied=["file_delete"],
                agent_depth=0,
            )

    def test_unlisted_tool_raises(self):
        with pytest.raises(ToolDenied, match="not in this agent's allowed tools"):
            validate_tool_access(
                tool_name="code_execute",
                agent_tools_allowed=["web_search"],
                agent_tools_denied=[],
                agent_depth=0,
            )

    def test_restricted_tool_blocked_at_depth_2(self):
        with pytest.raises(ToolDenied, match="restricted to agents at depth"):
            validate_tool_access(
                tool_name="shell_exec",
                agent_tools_allowed=["shell_exec"],
                agent_tools_denied=[],
                agent_depth=2,
            )

    def test_restricted_tool_allowed_at_depth_0(self):
        # Should not raise
        validate_tool_access(
            tool_name="shell_exec",
            agent_tools_allowed=["shell_exec"],
            agent_tools_denied=[],
            agent_depth=0,
        )

    def test_restricted_tool_allowed_at_depth_1(self):
        validate_tool_access(
            tool_name="shell_exec",
            agent_tools_allowed=["shell_exec"],
            agent_tools_denied=[],
            agent_depth=1,
        )


class TestGetToolDescriptions:
    def test_returns_requested_tools(self):
        tools = get_tool_descriptions(["web_search", "code_execute"])
        names = [t["function"]["name"] for t in tools]
        assert "web_search" in names
        assert "code_execute" in names

    def test_only_requested_tools_returned(self):
        tools = get_tool_descriptions(["web_search"])
        names = [t["function"]["name"] for t in tools]
        assert "web_search" in names
        assert "spawn_agent" not in names

    def test_spawn_agent_included_when_allowed(self):
        tools = get_tool_descriptions(["spawn_agent"])
        names = [t["function"]["name"] for t in tools]
        assert "spawn_agent" in names

    def test_empty_allowed_returns_nothing(self):
        tools = get_tool_descriptions([])
        assert len(tools) == 0

    def test_unknown_tool_skipped(self):
        tools = get_tool_descriptions(["nonexistent_tool"])
        names = [t["function"]["name"] for t in tools]
        assert "nonexistent_tool" not in names


@pytest.mark.asyncio
class TestExecuteTool:
    async def test_web_search_returns_results(self):
        result = await execute_tool(
            tool_name="web_search",
            args={"query": "test query"},
            agent_tools_allowed=["web_search"],
            agent_tools_denied=[],
            agent_depth=0,
        )
        assert result["tool"] == "web_search"
        assert "results" in result

    async def test_denied_tool_raises(self):
        with pytest.raises(ToolDenied):
            await execute_tool(
                tool_name="web_search",
                args={"query": "test"},
                agent_tools_allowed=["web_search"],
                agent_tools_denied=["web_search"],
                agent_depth=0,
            )

    async def test_unknown_tool_returns_error(self):
        result = await execute_tool(
            tool_name="made_up_tool",
            args={},
            agent_tools_allowed=["made_up_tool"],
            agent_tools_denied=[],
            agent_depth=0,
        )
        assert "error" in result
        assert "Unknown tool" in result["error"]
