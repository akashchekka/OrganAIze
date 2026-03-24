"""Tool registry — available tools + access control per agent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Awaitable

from config import OUTPUT_DIR, RESTRICTED_TOOLS, RESTRICTED_TOOLS_MAX_DEPTH

_WORKSPACE_ROOT = Path(OUTPUT_DIR).resolve().parent


class ToolDenied(Exception):
    """Raised when an agent tries to use a tool it's not allowed."""


# ── Tool Definitions ────────────────────────────────────────────
# Each tool is an async callable: (args: dict) -> dict

async def web_search(args: dict) -> dict:
    """Placeholder web search tool. In production, integrate with SerpAPI/Tavily/etc."""
    query = args.get("query", "")
    return {
        "tool": "web_search",
        "query": query,
        "results": [
            {"title": f"Result for: {query}", "snippet": "Placeholder result. Wire up a real search API."}
        ],
    }


async def code_execute(args: dict) -> dict:
    """Execute a Python code snippet in a sandboxed subprocess."""
    code = args.get("code", "")
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "tool": "code_execute",
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"tool": "code_execute", "error": "Execution timed out (30s)"}


def _is_within_workspace(path: Path) -> bool:
    """Check that a resolved path is within the workspace root."""
    try:
        path.resolve().relative_to(_WORKSPACE_ROOT)
        return True
    except ValueError:
        return False


async def file_read(args: dict) -> dict:
    """Read a file from the workspace."""
    filepath = args.get("path", "")
    path = Path(filepath)
    if not _is_within_workspace(path):
        return {"tool": "file_read", "error": f"Access denied: path outside workspace: {filepath}"}
    if not path.exists():
        return {"tool": "file_read", "error": f"File not found: {filepath}"}
    content = path.read_text(encoding="utf-8", errors="replace")[:10000]
    return {"tool": "file_read", "path": filepath, "content": content}


async def file_write(args: dict) -> dict:
    """Write content to a file."""
    filepath = args.get("path", "")
    content = args.get("content", "")
    path = Path(filepath)
    if not _is_within_workspace(path):
        return {"tool": "file_write", "error": f"Access denied: path outside workspace: {filepath}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"tool": "file_write", "path": filepath, "bytes_written": len(content)}


async def noop_tool(args: dict) -> dict:
    """Placeholder for tools not yet implemented."""
    return {"tool": "noop", "message": "This tool is not yet implemented."}


# ── Registry ────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[[dict], Awaitable[dict]]] = {
    "web_search": web_search,
    "code_execute": code_execute,
    "file_read": file_read,
    "file_write": file_write,
    "read_paper": noop_tool,
    "shell_exec": noop_tool,
    "file_delete": noop_tool,
    "network_request_external": noop_tool,
}


def validate_tool_access(
    tool_name: str,
    agent_tools_allowed: list[str],
    agent_tools_denied: list[str],
    agent_depth: int,
) -> None:
    """Check whether an agent is allowed to use a specific tool."""
    if tool_name in agent_tools_denied:
        raise ToolDenied(f"Tool '{tool_name}' is explicitly denied for this agent.")

    if tool_name not in agent_tools_allowed:
        raise ToolDenied(f"Tool '{tool_name}' is not in this agent's allowed tools.")

    if tool_name in RESTRICTED_TOOLS and agent_depth > RESTRICTED_TOOLS_MAX_DEPTH:
        raise ToolDenied(
            f"Tool '{tool_name}' is restricted to agents at depth <= {RESTRICTED_TOOLS_MAX_DEPTH} "
            f"(agent is at depth {agent_depth})."
        )


async def execute_tool(
    tool_name: str,
    args: dict,
    agent_tools_allowed: list[str],
    agent_tools_denied: list[str],
    agent_depth: int,
) -> dict:
    """Validate access and execute a tool."""
    validate_tool_access(tool_name, agent_tools_allowed, agent_tools_denied, agent_depth)

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return {"tool": tool_name, "error": f"Unknown tool: {tool_name}"}

    return await tool_fn(args)


def get_tool_descriptions(allowed_tools: list[str]) -> list[dict]:
    """Generate OpenAI-style tool/function descriptions for allowed tools."""
    descriptions = {
        "web_search": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information. Returns a list of results with titles and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        },
        "code_execute": {
            "type": "function",
            "function": {
                "name": "code_execute",
                "description": "Execute a Python code snippet and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                },
            },
        },
        "file_read": {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Path to the file"}},
                    "required": ["path"],
                },
            },
        },
        "file_write": {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        "spawn_agent": {
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Spawn a new sub-agent to handle a specific subtask. Only use when the task requires a different skill domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Agent name"},
                        "role": {"type": "string", "description": "Agent role: researcher|engineer|critic|qa|summarizer|pm|synthesizer"},
                        "persona": {"type": "string", "description": "One-line personality"},
                        "expertise": {"type": "array", "items": {"type": "string"}, "description": "Skill keywords"},
                        "task": {"type": "string", "description": "Clear task description"},
                        "success_criteria": {"type": "string", "description": "How to know the task is done"},
                        "output_format": {"type": "string", "description": "json|markdown|code|structured_report"},
                        "tools_needed": {"type": "array", "items": {"type": "string"}, "description": "Tools this agent needs"},
                        "parallel": {"type": "boolean", "description": "If true (default), this spawn runs in parallel with other spawns in the same turn. Set to false when this agent's output is needed before spawning the next."},
                    },
                    "required": ["name", "role", "task", "success_criteria"],
                },
            },
        },
    }

    result = []
    for tool in allowed_tools:
        if tool in descriptions:
            result.append(descriptions[tool])
    return result
