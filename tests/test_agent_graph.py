"""Tests for the LangGraph agent — graph structure and routing."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.agent import AgentState, build_agent_graph, should_continue


class TestShouldContinue:
    def test_returns_end_when_completed(self):
        state: AgentState = {
            "messages": [],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": [], "tools_denied": [],
            "final_output": "done", "step_count": 5, "max_steps": 10,
            "status": "completed",
        }
        assert should_continue(state) == "end"

    def test_returns_end_when_expired(self):
        state: AgentState = {
            "messages": [],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": [], "tools_denied": [],
            "final_output": "", "step_count": 5, "max_steps": 10,
            "status": "expired",
        }
        assert should_continue(state) == "end"

    def test_returns_end_when_budget_exhausted(self):
        state: AgentState = {
            "messages": [],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": [], "tools_denied": [],
            "final_output": "", "step_count": 5, "max_steps": 10,
            "status": "token_cap_exceeded",
        }
        assert should_continue(state) == "end"

    def test_returns_tools_when_tool_calls(self):
        ai_msg = AIMessage(
            content="I need to search",
            tool_calls=[{"id": "tc1", "name": "web_search", "args": {"query": "test"}}],
        )
        state: AgentState = {
            "messages": [ai_msg],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": ["web_search"], "tools_denied": [],
            "final_output": "", "step_count": 5, "max_steps": 10,
            "status": "running",
        }
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(self):
        ai_msg = AIMessage(content="Here is my final answer")
        state: AgentState = {
            "messages": [ai_msg],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": [], "tools_denied": [],
            "final_output": "", "step_count": 5, "max_steps": 10,
            "status": "running",
        }
        assert should_continue(state) == "end"

    def test_returns_end_for_human_message(self):
        state: AgentState = {
            "messages": [HumanMessage(content="hello")],
            "agent_id": "a1", "agent_name": "A1", "session_id": "s1",
            "depth": 0, "model": "gpt-4o",
            "tools_allowed": [], "tools_denied": [],
            "final_output": "", "step_count": 0, "max_steps": 10,
            "status": "running",
        }
        assert should_continue(state) == "end"


class TestBuildAgentGraph:
    def test_graph_has_correct_nodes(self):
        graph = build_agent_graph()
        node_names = set(graph.nodes.keys())
        assert "reason" in node_names
        assert "tools" in node_names

    def test_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_agent_graph()
        app = graph.compile(checkpointer=MemorySaver())
        assert app is not None
