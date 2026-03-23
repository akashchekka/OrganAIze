"""LangGraph-based Agent — each agent is a reactive graph with tool-calling.

Architecture:
    reason (LLM) --> should_continue --> tools --> reason (loop)
                                     --> END

Each agent instance compiles into its own LangGraph StateGraph.
Spawned children are themselves full graphs, invoked recursively.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from langchain_core.runnables import RunnableConfig

import litellm

from config import MAX_CONCURRENT_LLM_CALLS, get_llm_kwargs
from core.blueprint import AgentBlueprint, compile_system_prompt
from core.cost_tracker import TokenCapExceeded, CostTracker
from tools.tool_registry import ToolDenied, execute_tool, get_tool_descriptions

logger = logging.getLogger("organaize.agent")

_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# ── Agent State ─────────────────────────────────────────────────

class AgentState(TypedDict):
    """State flowing through the agent's LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    agent_id: str
    agent_name: str
    session_id: str
    depth: int
    model: str
    tools_allowed: list[str]
    tools_denied: list[str]
    final_output: str
    step_count: int
    max_steps: int
    status: str  # "running" | "completed" | "expired" | "budget_exhausted"


# ── Node: Reason ────────────────────────────────────────────────

async def reason_node(state: AgentState, config: RunnableConfig) -> dict:
    """Call the LLM to reason about the next step."""
    deps = config["configurable"]
    cost_tracker: CostTracker = deps["cost_tracker"]

    # Check step limit
    if state["step_count"] >= state["max_steps"]:
        logger.warning("Agent %s hit max steps (%d)", state["agent_name"], state["max_steps"])
        return {"status": "completed", "final_output": f"[MAX_STEPS] Reached {state['max_steps']} steps."}

    # Rate-limited LLM call
    tools = get_tool_descriptions(state["tools_allowed"])
    async with _llm_semaphore:
        kwargs = get_llm_kwargs(state["model"])
        kwargs["messages"] = [_to_litellm_msg(m) for m in state["messages"]]
        kwargs["temperature"] = 0.4
        if tools:
            kwargs["tools"] = tools

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            logger.error("LLM call failed for %s: %s", state["agent_name"], e)
            return {"status": "completed", "final_output": f"[LLM_ERROR] {e}"}

    # Track tokens
    usage = response.usage
    if usage:
        try:
            cost_tracker.charge(
                agent_id=state["agent_id"],
                model=state["model"],
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        except TokenCapExceeded as e:
            logger.warning("Token cap exceeded for %s: %s", state["agent_name"], e)
            return {"status": "token_cap_exceeded", "final_output": f"[TOKEN_CAP_EXCEEDED] {e}"}

    msg = response.choices[0].message

    # Convert to LangChain message
    if msg.tool_calls:
        ai_msg = AIMessage(
            content=msg.content or "",
            tool_calls=[
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                }
                for tc in msg.tool_calls
            ],
        )
    else:
        ai_msg = AIMessage(content=msg.content or "")

    updates: dict[str, Any] = {
        "messages": [ai_msg],
        "step_count": state["step_count"] + 1,
    }

    if not msg.tool_calls:
        updates["final_output"] = msg.content or ""
        updates["status"] = "completed"

    return updates


# ── Node: Tools ─────────────────────────────────────────────────

async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last AI message."""
    deps = config["configurable"]
    spawner = deps["spawner"]
    blueprint: AgentBlueprint = deps["blueprint"]

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    # Separate spawn calls from other tools
    spawn_calls = []
    other_calls = []
    for tc in last_msg.tool_calls:
        if tc["name"] == "spawn_agent":
            spawn_calls.append(tc)
        else:
            other_calls.append(tc)

    tool_messages = []

    # Run non-spawn tools sequentially (order may matter)
    for tc in other_calls:
        result = await _execute_single_tool(tc, state, deps)
        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    # Split spawns by parallel flag (LLM decides)
    sequential_spawns = [tc for tc in spawn_calls if not tc["args"].get("parallel", True)]
    parallel_spawns = [tc for tc in spawn_calls if tc["args"].get("parallel", True)]

    # Run sequential spawns first, in order
    for tc in sequential_spawns:
        result = await _run_spawn_and_log(tc, state, deps, blueprint, spawner)
        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    # Fan out parallel spawns
    if parallel_spawns:
        tasks = [
            _run_spawn_and_log(tc, state, deps, blueprint, spawner)
            for tc in parallel_spawns
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for tc, result in zip(parallel_spawns, results):
            if isinstance(result, Exception):
                logger.error("Parallel spawn failed for %s: %s", tc["args"].get("name"), result)
                result = {"tool": "spawn_agent", "error": str(result)}
            tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    return {"messages": tool_messages}


async def _execute_single_tool(tc: dict, state: AgentState, deps: dict) -> dict:
    """Execute a single non-spawn tool call."""
    tool_name = tc["name"]
    args = tc["args"]

    try:
        result = await execute_tool(
            tool_name=tool_name,
            args=args,
            agent_tools_allowed=state["tools_allowed"],
            agent_tools_denied=state["tools_denied"],
            agent_depth=state["depth"],
        )
    except (ToolDenied, Exception) as e:
        result = {"tool": tool_name, "error": str(e)}
        logger.error("Tool %s failed for %s: %s", tool_name, state["agent_name"], e)

    return result


async def _run_spawn_and_log(tc: dict, state: AgentState, deps: dict, blueprint: AgentBlueprint, spawner) -> dict:
    """Execute a spawn_agent tool call."""
    args = tc["args"]

    try:
        result = await _handle_spawn(args, blueprint, spawner, deps)
    except Exception as e:
        result = {"tool": "spawn_agent", "error": str(e)}
        logger.error("spawn_agent failed for %s: %s", state["agent_name"], e)

    return result


# ── Routing ─────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route: if agent produced a final output or hit limits → end. Otherwise → tools."""
    if state.get("status") in ("completed", "expired", "token_cap_exceeded"):
        return "end"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    return "end"


# ── Spawn Handler ───────────────────────────────────────────────

async def _handle_spawn(args: dict, parent_bp: AgentBlueprint, spawner, deps: dict) -> dict:
    """Spawn a child agent and run its own LangGraph to completion."""
    child_bp = spawner.spawn(parent_bp, args)
    if not child_bp:
        logger.info("Spawn denied for parent %s (budget exhausted)", parent_bp.name)
        return {"tool": "spawn_agent", "status": "denied", "reason": "Budget exhausted or global cap reached."}

    logger.info("Spawning child %s (role=%s, depth=%d) from parent %s",
                child_bp.name, child_bp.role, child_bp.depth, parent_bp.name)

    child_output = await run_agent_graph(
        blueprint=child_bp,
        session_id=deps["session_id"],
        cost_tracker=deps["cost_tracker"],
        spawner=deps["spawner"],
        checkpointer=deps.get("checkpointer"),
    )

    summary = child_output[:3000] if len(child_output) <= 3000 else child_output[:3000] + "\n...[truncated]"
    logger.info("Child %s completed (output_len=%d)", child_bp.name, len(child_output))
    return {"tool": "spawn_agent", "status": "completed", "agent_name": child_bp.name, "output": summary}


# ── Graph Builder ───────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Build the LangGraph for a single agent.

    Graph topology:
        reason ──► should_continue ──► tools ──► reason (loop)
                                   └──► END
    """
    graph = StateGraph(AgentState)

    graph.add_node("reason", reason_node)
    graph.add_node("tools", tool_node, retry_policy=RetryPolicy(max_attempts=3))

    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "reason")

    return graph


# ── Runner ──────────────────────────────────────────────────────

async def run_agent_graph(
    blueprint: AgentBlueprint,
    session_id: str,
    cost_tracker: CostTracker,
    spawner,
    checkpointer=None,
    initial_context: str = "",
) -> str:
    """Create and run a LangGraph agent for the given blueprint.

    Returns the agent's final output string.
    """
    logger.info("Starting agent graph: %s (model=%s, depth=%d)",
                blueprint.name, blueprint.model, blueprint.depth)
    system_prompt = compile_system_prompt(blueprint)

    # Build initial messages
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if initial_context:
        messages.append(HumanMessage(content=initial_context))
    messages.append(HumanMessage(content=f"Execute your task now:\n\n{blueprint.task}"))

    initial_state: AgentState = {
        "messages": messages,
        "agent_id": blueprint.agent_id,
        "agent_name": blueprint.name,
        "session_id": session_id,
        "depth": blueprint.depth,
        "model": blueprint.model,
        "tools_allowed": blueprint.tools_allowed,
        "tools_denied": blueprint.tools_denied,
        "final_output": "",
        "step_count": 0,
        "max_steps": blueprint.max_steps,
        "status": "running",
    }

    cp = checkpointer or MemorySaver()
    graph = build_agent_graph()
    app = graph.compile(checkpointer=cp)

    config = {
        "configurable": {
            "thread_id": blueprint.agent_id,
            "cost_tracker": cost_tracker,
            "blueprint": blueprint,
            "spawner": spawner,
            "session_id": session_id,
            "checkpointer": cp,
        }
    }

    final_state = await app.ainvoke(initial_state, config=config)

    output = final_state.get("final_output", "")
    status = final_state.get("status", "completed")
    logger.info("Agent %s finished (status=%s, output_len=%d)", blueprint.name, status, len(output))

    return output


# ── Helpers ─────────────────────────────────────────────────────

def _to_litellm_msg(msg: BaseMessage) -> dict:
    """Convert a LangChain message to a litellm-compatible dict."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                }
                for tc in msg.tool_calls
            ]
        return d
    elif isinstance(msg, ToolMessage):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    return {"role": "user", "content": str(msg.content)}
