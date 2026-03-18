"""LangGraph-based Agent — each agent is a reactive graph with tool-calling.

Architecture:
    reason (LLM) ──► should_continue ──► tools ──► reason (loop)
                                     └──► END

Each agent instance compiles into its own LangGraph StateGraph.
Spawned children are themselves full graphs, invoked recursively.
Checkpointing via LangGraph's MemorySaver enables hibernate/resume.
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

import litellm

from config import MAX_CONCURRENT_LLM_CALLS, get_llm_kwargs
from core.blueprint import AgentBlueprint, compile_system_prompt
from core.cost_tracker import BudgetExhausted, CostTracker
from core.lifecycle import AgentExpired, AgentLifecycle
from tracing.event_logger import EventLogger
from tracing.event_types import (
    AGENT_OUTPUT,
    AGENT_THOUGHT,
    AGENT_TOOL_CALL,
    AGENT_TOOL_ERROR,
)
from memory.blackboard import BlackboardClient
from tools.tool_registry import ToolDenied, execute_tool, get_tool_descriptions

logger = logging.getLogger("evolve.agent")

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

async def reason_node(state: AgentState, *, config: dict) -> dict:
    """Call the LLM to reason about the next step."""
    deps = config["configurable"]
    cost_tracker: CostTracker = deps["cost_tracker"]
    event_logger: EventLogger = deps["event_logger"]
    lifecycle: AgentLifecycle = deps["lifecycle"]

    # Check TTL + steps
    try:
        await lifecycle.step()
    except AgentExpired as e:
        logger.warning("Agent %s expired: %s", state["agent_name"], e)
        return {"status": "expired", "final_output": f"[EXPIRED] {e}"}

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

    # Track cost
    usage = response.usage
    if usage:
        try:
            await cost_tracker.charge(
                agent_id=state["agent_id"],
                agent_name=state["agent_name"],
                depth=state["depth"],
                model=state["model"],
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        except BudgetExhausted as e:
            logger.warning("Budget exhausted for %s: %s", state["agent_name"], e)
            return {"status": "budget_exhausted", "final_output": f"[BUDGET_EXHAUSTED] {e}"}

    msg = response.choices[0].message

    # Log thought
    if msg.content:
        await event_logger.log_event(
            session_id=state["session_id"],
            agent_id=state["agent_id"],
            agent_name=state["agent_name"],
            depth=state["depth"],
            event_type=AGENT_THOUGHT,
            payload={"thought_text": msg.content[:2000]},
        )

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

async def tool_node(state: AgentState, *, config: dict) -> dict:
    """Execute tool calls from the last AI message."""
    deps = config["configurable"]
    event_logger: EventLogger = deps["event_logger"]
    blackboard: BlackboardClient = deps["blackboard"]
    spawner = deps["spawner"]
    blueprint: AgentBlueprint = deps["blueprint"]

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    tool_messages = []
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        args = tc["args"]

        try:
            if tool_name == "share_finding":
                await blackboard.share(args.get("key", "unknown"), args.get("value", ""))
                result = {"tool": "share_finding", "status": "shared", "key": args.get("key")}

            elif tool_name == "read_shared":
                shared = await blackboard.read_shared(args.get("pattern", "*"))
                simplified = {k.split(":")[-1]: v for k, v in shared.items()}
                result = {"tool": "read_shared", "findings": simplified}

            elif tool_name == "spawn_agent":
                result = await _handle_spawn(args, blueprint, spawner, deps)

            else:
                result = await execute_tool(
                    tool_name=tool_name,
                    args=args,
                    agent_tools_allowed=state["tools_allowed"],
                    agent_tools_denied=state["tools_denied"],
                    agent_depth=state["depth"],
                )

            await event_logger.log_event(
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_name=state["agent_name"],
                depth=state["depth"],
                event_type=AGENT_TOOL_CALL,
                payload={"tool_name": tool_name, "args": args, "result_preview": str(result)[:1000]},
            )

        except (ToolDenied, Exception) as e:
            result = {"tool": tool_name, "error": str(e)}
            logger.error("Tool %s failed for %s: %s", tool_name, state["agent_name"], e)
            await event_logger.log_event(
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_name=state["agent_name"],
                depth=state["depth"],
                event_type=AGENT_TOOL_ERROR,
                payload={"tool_name": tool_name, "error": str(e)},
            )

        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    return {"messages": tool_messages}


# ── Routing ─────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route: if agent produced a final output or hit limits → end. Otherwise → tools."""
    if state.get("status") in ("completed", "expired", "budget_exhausted"):
        return "end"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    return "end"


# ── Spawn Handler ───────────────────────────────────────────────

async def _handle_spawn(args: dict, parent_bp: AgentBlueprint, spawner, deps: dict) -> dict:
    """Spawn a child agent and run its own LangGraph to completion."""
    child_bp = await spawner.maybe_spawn(parent_bp, args)
    if not child_bp:
        logger.info("Spawn denied for parent %s (reuse or budget)", parent_bp.name)
        return {"tool": "spawn_agent", "status": "denied", "reason": "Reuse existing or budget exhausted."}

    logger.info("Spawning child %s (role=%s, depth=%d) from parent %s",
                child_bp.name, child_bp.role, child_bp.depth, parent_bp.name)

    child_output = await run_agent_graph(
        blueprint=child_bp,
        session_id=deps["session_id"],
        cost_tracker=deps["cost_tracker"],
        event_logger=deps["event_logger"],
        blackboard_redis=deps["blackboard"].redis,
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
    graph.add_node("tools", tool_node)

    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "reason")

    return graph


# ── Runner ──────────────────────────────────────────────────────

async def run_agent_graph(
    blueprint: AgentBlueprint,
    session_id: str,
    cost_tracker: CostTracker,
    event_logger: EventLogger,
    blackboard_redis,
    spawner,
    checkpointer=None,
    initial_context: str = "",
) -> str:
    """Create and run a LangGraph agent for the given blueprint.

    - Each agent gets its own compiled graph with a thread_id = agent_id.
    - MemorySaver checkpointing enables pause/resume (hibernate/resurrect).
    - Spawned children recursively call this function, creating nested graphs.

    Returns the agent's final output string.
    """
    logger.info("Starting agent graph: %s (model=%s, depth=%d, ttl=%ds)",
                blueprint.name, blueprint.model, blueprint.depth, blueprint.ttl_seconds)
    system_prompt = compile_system_prompt(blueprint)

    lifecycle = AgentLifecycle(
        agent_id=blueprint.agent_id,
        agent_name=blueprint.name,
        depth=blueprint.depth,
        ttl_seconds=blueprint.ttl_seconds,
        max_steps=blueprint.max_steps,
        session_id=session_id,
        registry=spawner.registry,
        event_logger=event_logger,
        redis=blackboard_redis,
    )
    await lifecycle.start()

    blackboard = BlackboardClient(
        agent_id=blueprint.agent_id,
        session_id=session_id,
        redis=blackboard_redis,
    )

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

    # Compile graph with checkpointer
    cp = checkpointer or MemorySaver()
    graph = build_agent_graph()
    app = graph.compile(checkpointer=cp)

    config = {
        "configurable": {
            "thread_id": blueprint.agent_id,
            "cost_tracker": cost_tracker,
            "event_logger": event_logger,
            "lifecycle": lifecycle,
            "blueprint": blueprint,
            "blackboard": blackboard,
            "spawner": spawner,
            "session_id": session_id,
            "checkpointer": cp,
        }
    }

    # Run the graph
    final_state = await app.ainvoke(initial_state, config=config)

    # Post-completion handling
    output = final_state.get("final_output", "")
    status = final_state.get("status", "completed")

    if status == "completed":
        await lifecycle.complete(output)
        logger.info("Agent %s completed successfully (output_len=%d)", blueprint.name, len(output))
        if blueprint.memory_scope.share_policy in ("auto_conclusions", "full_transparency"):
            await blackboard.share(
                f"output:{blueprint.name}",
                {"agent": blueprint.name, "output": output[:5000]},
            )
    elif status == "expired":
        pass  # lifecycle already handled in reason_node
    elif status == "budget_exhausted":
        await lifecycle.die(reason="budget_exhausted")
        logger.warning("Agent %s died: budget exhausted", blueprint.name)

    await event_logger.log_event(
        session_id=session_id,
        agent_id=blueprint.agent_id,
        agent_name=blueprint.name,
        depth=blueprint.depth,
        event_type=AGENT_OUTPUT,
        payload={"output": output[:5000], "status": status},
    )

    await cost_tracker.rollup_subtree(blueprint.agent_id)
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
