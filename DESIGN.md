# OrganAIze — Self-Organizing Agentic AI

> An architecture where agents self-organize to solve a problem, just like humans do.

---

## High-Level Architecture

```
                        ┌───────────────────────┐
                        │     User Request      │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │    Genesis Agent      │
                        │   (The Orchestrator)  │
                        │   depth=0, budget=N   │
                        └──┬────────┬────────┬──┘
                           │        │        │
                  ┌────────▼──┐ ┌───▼───┐ ┌──▼────────┐
                  │  Agent A  │ │Agent B│ │  Agent C  │
                  │  depth=1  │ │depth=1│ │  depth=1  │
                  └──┬─────┬──┘ └───────┘ └───────────┘
                     │     │
               ┌─────▼──┐ ┌▼──────┐
               │ Sub A1  │ │Sub A2 │
               │ depth=2 │ │depth=2│
               └─────────┘ └──────┘
```

**No databases. No Redis. No MongoDB.** Everything runs in-memory within a single process. Session output is written to disk as JSON + Markdown files.

### Infrastructure

| Component | Tech | Purpose |
|---|---|---|
| Agent runtime | LangGraph `StateGraph` | `reason → tools → reason` loop per agent |
| LLM calls | LiteLLM | Unified API across OpenAI, Azure, Anthropic, Ollama |
| Token tracking | In-memory `CostTracker` | Per-agent accounting, session-wide cap |
| Spawning | In-memory `Spawner` | Budget enforcement, depth/global caps |
| Checkpointing | LangGraph `MemorySaver` | In-memory state snapshots per agent |
| Output | File system | `output/<session_id>/session.json` + `output.md` |

---

## 1. Spawning — How Agents Decide and Divide

An agent shouldn't spawn because it *can*, but because it *must*. Two layers control this:

### LLM-Driven Decomposition

The LLM decides *whether* and *how* to spawn. When Genesis (or any agent with spawn permission) receives a task, it reasons about the goal and emits `spawn_agent` tool calls for subtasks that need different expertise. The LLM sees the `spawn_agent` tool in its tool list with parameters for name, role, task, expertise, and `parallel` flag — and decides naturally through the `reason → tools` loop.

Each `spawn_agent` call includes a `parallel` flag (default: `true`). Multiple spawns in a single LLM response with `parallel: true` fan out via `asyncio.gather`. Setting `parallel: false` runs them sequentially. The LLM can also spawn one agent per turn for fully sequential execution — just emit one `spawn_agent`, get the result, then decide the next.

*See: `core/agent.py` (`tool_node`), `tools/tool_registry.py`*

### Spawn Budget (Solves: Infinite Spawning)

The system decides *if it's allowed*. Every agent is born with a **spawn budget** inherited from its parent. Three env-configurable limits control it: `ORGANAIZE_MAX_AGENTS` (default: 20, global cap), `ORGANAIZE_MAX_DEPTH` (default: 4), and `ORGANAIZE_GENESIS_BUDGET` (default: 8, genesis's max children).

Inside `spawner.spawn()`, the child receives a `SpawnBudget` with `max_children` halved from the parent, `max_depth_remaining` decremented by 1, and `remaining_global` set to the current headroom.

*See: `core/spawner.py`, `core/blueprint.py` (`SpawnBudget`), `config.py`*

Rules:
- **Depth limit = 4.** No agent at depth 4 can spawn.
- **Budget halving.** A depth-0 agent with budget 8 gives each child budget 4, grandchildren budget 2, etc.
- **Global cap = 20.** Once 20 agents exist, no spawning allowed regardless of budget.

---

## 2. Agent Trait Generation — The DNA Blueprint

Every agent is defined by an **AgentBlueprint** — a structured document that the parent agent generates and the system interprets:

### The AgentBlueprint Schema

`AgentBlueprint` is a dataclass that defines everything about an agent:

- **Identity** — `agent_id` (UUID), `name`, `lineage` (ancestry chain)
- **Persona** — `role` (researcher/engineer/critic/etc.), `persona` (one-line personality), `expertise` (skill keywords), `verbosity`, `risk_tolerance`
- **Capabilities** — `tools_allowed`, `tools_denied`, `model` (auto-selected from role if not set)
- **Limits** — `max_steps`, `spawn_budget`
- **Task** — `task`, `success_criteria`, `output_format`

*See: `core/blueprint.py`*

### System Prompt Generation

The parent agent doesn't write raw system prompts. It fills in the `AgentBlueprint`, and a **prompt compiler** (`compile_system_prompt()`) turns it into a system prompt string.

The compiler checks if `spawn_agent` is in `tools_allowed` and injects either `SPAWN_ALLOWED_SECTION` (with budget info) or `SPAWN_DENIED_SECTION`. Then it formats `AGENT_SYSTEM_PROMPT_TEMPLATE` with all blueprint fields: name, role, persona, expertise, task, success criteria, tools, and risk tolerance.

*See: `core/blueprint.py` (`compile_system_prompt`), `prompts/agent.py`*

### Model Selection by Role (Solves: Cost Explosion)

Not every agent needs the most expensive model:

| Role | Default Model | Rationale |
|---|---|---|
| Genesis / Orchestrator | azure/gpt-4o | Needs best reasoning for decomposition |
| Researcher | azure/gpt-4o | Needs good synthesis of search results |
| Engineer | azure/gpt-4o | Good at code generation |
| Critic / QA | azure/gpt-4o-mini | Checking is cheaper than creating |
| Summarizer | azure/gpt-4o-mini | Compression task, lightweight |
| PM / Tracker | azure/gpt-4o-mini | Status tracking, minimal reasoning |
| Synthesizer | azure/gpt-4o | Needs strong reasoning for synthesis |

Model routing is handled by **LiteLLM**, which auto-detects the provider from the model prefix (e.g., `azure/gpt-4o` → Azure OpenAI, `gpt-4o` → OpenAI, `claude-sonnet-4-20250514` → Anthropic). Defaults are configured in `config.py` via `ROLE_MODEL_DEFAULTS`.

The parent agent can override this, but defaults keep costs bounded.

---

## 3. Communication Model — Direct Output Flow

Agents communicate through **direct parent-child tool messages**. No shared memory, no pub/sub, no blackboard.

### How It Works

When a parent agent calls `spawn_agent`, the child runs to completion and returns its output as a `ToolMessage`. The parent's LLM sees these outputs in its conversation history and can:

- Synthesize them into a final response
- Spawn additional agents based on what it learned
- Request more detail by spawning follow-up agents

### Context Aggregation (Solves: Context Window Blow-up)

When child agents complete, their output is returned to the parent via `_handle_spawn()` in the tool node. Each child's output is truncated to 3000 characters before being wrapped in a `ToolMessage`. If the output is shorter, it's passed as-is.

*See: `core/agent.py` (`_handle_spawn`)*

---

## 4. Addressing the Hard Problems

### Infinite Spawning → Budget + Depth + Global Cap

Three independent layers of defense. No single agent can circumvent all three.

```
Spawn request arrives at spawner.spawn()
    │
    ├── 1. GLOBAL check: agents_created < MAX_GLOBAL_AGENTS?  → NO → denied
    ├── 2. BUDGET check: parent.spawn_budget.can_spawn?        → NO → denied
    └── All pass → spawn allowed
```

**Layer 1: Depth Limit (vertical control)**

Prevents infinite nesting. Each child's `max_depth_remaining` = parent's - 1. The `SpawnBudget.can_spawn` property checks that `max_children > 0`, `max_depth_remaining > 0`, and `remaining_global > 0` — all three must pass.

At depth = `MAX_DEPTH` (4), agents get `spawn_agent` added to `tools_denied` — the LLM never even sees the option.

*See: `core/blueprint.py` (`SpawnBudget`), `core/spawner.py`*

**Layer 2: Budget Halving (horizontal control)**

Prevents sibling explosion. Each level gets half the parent's budget:

| Depth | max_children | Calculation |
|---|---|---|
| 0 (Genesis) | 8 | `GENESIS_SPAWN_BUDGET` |
| 1 | 4 | `8 // 2` |
| 2 | 2 | `4 // 2` |
| 3 | 1 | `2 // 2` |
| 4 | 0 | Cannot spawn |

Each successful spawn decrements the parent's budget in memory.

*See: `core/spawner.py`*

**Layer 3: Global Cap (absolute control)**

Regardless of individual budgets, once `MAX_GLOBAL_AGENTS` (default: 20) agents have been created in the session, all spawning stops. The `Spawner` tracks `agents_created` as a simple counter.

*See: `core/spawner.py`*

### Context Window Blow-up → Inline Output Truncation

Child agent outputs are returned to the parent as tool results. Each child output is truncated to 3000 chars in `_handle_spawn()`.

### Tool Safety → Allowlist per Agent

Tools are allowlisted, not blocklisted. Each agent only gets the tools its blueprint specifies — the LLM never sees tools it's not authorized to use.

Three checks run before every tool execution:

1. **Deny list** — if the tool is in `tools_denied`, it's rejected immediately (e.g., `spawn_agent` for depth-4 agents).
2. **Allow list** — if the tool isn't in `tools_allowed`, it's rejected. Agents can't discover tools outside their blueprint.
3. **Restricted tools** — dangerous tools (`shell_exec`, `file_delete`, `network_request_external`) are only available to agents at depth <= `RESTRICTED_TOOLS_MAX_DEPTH` (default: 1). Even if a blueprint lists them, deeper agents are blocked.

*See: `tools/tool_registry.py`*

### Cost Explosion → Token Budget + Model Tiering + Per-Agent Accounting

Two classes in `core/cost_tracker.py` handle all token accounting:

**`TokenUsage`** — A simple dataclass ledger that tracks `input_tokens`, `output_tokens`, `total_tokens`, and `llm_calls` for a single agent. Updated via `record(inp, out)` after each LLM call.

**`CostTracker`** — Session-wide singleton shared by all agents in a run:

| Method | When Called | What It Does |
|---|---|---|
| `charge()` | After every LLM call | Adds tokens to in-memory `TokenUsage` for that agent. Raises `TokenCapExceeded` if session total hits `MAX_SESSION_TOKENS` (default: 500k). |
| `get_session_summary()` | At session end | Returns full breakdown: per-agent totals, session totals, `tokens_remaining`. |

Model tiering (cheap models for cheap tasks) keeps costs bounded — see Section 2. LLM concurrency is capped at `MAX_CONCURRENT_LLM_CALLS` (default: 3) via an asyncio semaphore in `core/agent.py`.

### Deadlocks → No Circular Dependencies by Design

Information flows DOWN (parent → child via `spawn_agent`) and UP (child → parent via `ToolMessage`). Agents at the same depth never wait on each other — they run independently and return results to their parent.

---

## 5. Putting It All Together — Full Flow

Based on an actual test run:

```
1. User submits: "Build a project plan: research Python web frameworks,
   design a REST API for a todo app, write a security checklist"

2. Genesis Agent receives task (LangGraph StateGraph)
   ├── reason_node: LLM analyzes the goal, identifies 3 independent domains
   ├── Emits 3 spawn_agent tool calls in a single response (all parallel: true)
   ├── should_continue → routes to "tools"
   └── tool_node separates spawns, fans out via asyncio.gather

3. Three child agents spawn IN PARALLEL:
   ├── PythonWebFrameworksResearcher (role=researcher, depth=1)
   │   ├── Uses web_search to find framework comparisons
   │   └── Completes in 5s (1,415 tokens)
   │
   ├── TodoAppRestApiDesigner (role=engineer, depth=1)
   │   ├── Designs REST endpoints, schemas, auth flow
   │   └── Completes in 8s (1,609 tokens)
   │
   └── SecurityChecklistWriter (role=critic, depth=1)
       ├── Produces deployment security checklist
       └── Completes in 9s (1,716 tokens)

4. Genesis Agent graph continues (reason_node receives all 3 ToolMessages):
   ├── Reads child outputs (each truncated to 3000 chars)
   ├── reason_node synthesizes all three into a cohesive project plan
   ├── should_continue → END (no more tool calls)
   └── Completes (genesis tokens=7,419)

5. Session output written to output/<session_id>/:
   ├── session.json — metadata, token summary, agent count
   └── output.md — final synthesized markdown
   
   4 agents total, 12,159 tokens, 7 LLM calls, 22 seconds wall time
```

---

## 6. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Agent Runtime** | LangGraph `StateGraph` | Each agent is a `reason → tools → reason` graph. Dynamic node creation for spawned agents. Built-in checkpointing. |
| **LLM Calls** | LiteLLM | Unified API across OpenAI, Anthropic, Azure, Ollama — easy model tiering |
| **Checkpointing** | LangGraph `MemorySaver` | In-memory state persistence per agent. Enables crash recovery within a session. |
| **Token Tracking** | `CostTracker` | Pure in-memory per-agent ledger. Session cap via `TokenCapExceeded`. |
| **Spawning** | `Spawner` | Pure in-memory. Budget enforcement, depth/global caps. Single-process — no locks needed. |
| **Output** | File system | `output/<session_id>/session.json` + `output.md`. Simple, inspectable, version-controllable. |
| **Orchestration** | asyncio + semaphore | Parallel agent execution with concurrency cap |

---

## 7. Project Structure

```
OrganAIze/
├── core/
│   ├── genesis.py          # Genesis Agent — orchestrator + file output
│   ├── agent.py            # LangGraph StateGraph: reason → tools → reason loop
│   │                        #   AgentState, reason_node, tool_node, should_continue
│   │                        #   build_agent_graph(), run_agent_graph()
│   ├── blueprint.py        # AgentBlueprint, SpawnBudget + prompt compiler
│   ├── spawner.py          # Spawn logic: budget enforcement, agent creation
│   └── cost_tracker.py     # Token-based budget tracking
│
├── prompts/
│   ├── genesis.py          # GENESIS_PERSONA
│   └── agent.py            # AGENT_SYSTEM_PROMPT_TEMPLATE, SPAWN_ALLOWED/DENIED
│
├── tools/
│   └── tool_registry.py    # Tool definitions, access control, OpenAI-format specs
│
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_agent_graph.py
│   ├── test_blueprint.py
│   ├── test_spawn_modes.py
│   ├── test_spawner.py
│   └── test_tool_registry.py
│
├── site/
│   ├── index.html          # Landing page
│   └── style.css           # Site styles
│
├── config.py               # Global caps, defaults, model routing, LiteLLM kwargs
├── main.py                 # CLI entry point
├── pyproject.toml          # Project metadata
├── requirements.txt
├── README.md
└── DESIGN.md               # This document
```

### LangGraph Agent Architecture

Each agent (including Genesis) runs as a LangGraph `StateGraph`:

```
┌───────────────────────────────────────────────────────┐
│                 Agent StateGraph                      │
│                                                       │
│  ┌────────┐   should_continue   ┌────────┐            │
│  │ reason ├──► has tool calls? ─► tools  ├────┐       │
│  │ (LLM)  │   │                 │ (exec) │    │       │
│  └────┬───┘   │                 └────────┘    │       │
│       ▲       │ no tool calls                 │       │
│       └───────┴───────────────────────────────┘       │
│               │                                       │
│               ▼                                       │
│          ┌───────┐                                    │
│          │  END  │                                    │
│          └───────┘                                    │
│                                                       │
│  When tool = spawn_agent:                             │
│    tool_node creates a NEW StateGraph for the child   │
│    and runs it to completion.                         │
│    Output returned as ToolMessage.                    │
│                                                       │
│  Checkpointer: MemorySaver (thread_id = agent_id)     │
└───────────────────────────────────────────────────────┘
```

### Dynamic Spawning — How It Works in LangGraph

The graph topology is **structurally static** (`reason → tools → reason`) but **dynamically recursive** — the `tool_node` can create entirely new agent graphs at runtime.

**Step-by-step flow:**

```
1. reason_node (Parent)
   │  LLM decides what to do. Emits AIMessage with tool calls:
   │  spawn_agent({name: "Researcher", role: "researcher", task: "...", parallel: true})
   │  spawn_agent({name: "Designer", role: "engineer", task: "...", parallel: true})
   │  spawn_agent({name: "Reviewer", role: "critic", task: "...", parallel: true})
   │
   ▼
2. should_continue
   │  Sees tool_calls on AIMessage → routes to "tools"
   │
   ▼
3. tool_node (Parent)
   │  Separates spawn calls from non-spawn tools.
   │  Runs non-spawn tools sequentially.
   │  Then splits spawns by parallel flag:
   │    - parallel: false → run one at a time, in order
   │    - parallel: true  → fan out via asyncio.gather
   │
   │  For each spawn → calls _run_spawn_and_log() → _handle_spawn():
   │
   ▼
4. _handle_spawn()
   │  ├── spawner.spawn() →
   │  │     ├── Checks global cap (agents_created < MAX_GLOBAL_AGENTS)
   │  │     ├── Checks parent budget (can_spawn?)
   │  │     ├── Creates child AgentBlueprint
   │  │     └── Decrements parent budget, increments counter
   │  │
   │  └── run_agent_graph(child_blueprint, ...) →
   │        Creates a NEW StateGraph for the child:
   │        reason ──► tools ──► reason ──► END
   │        Child runs to completion, returns output string.
   │        (Child can also spawn its own children recursively.)
   │
   ▼
5. tool_node (Parent) — continued
   │  Collects all child outputs (truncated to 3000 chars each).
   │  Wraps each in a ToolMessage:
   │  ToolMessage(content='{"tool": "spawn_agent", "status": "completed",
   │      "agent_name": "Researcher", "output": "...findings..."}')
   │
   ▼
6. reason_node (Parent)
   │  LLM sees all ToolMessages with child outputs.
   │  Can now:
   │    - Spawn more agents (another round of tool calls)
   │    - Synthesize the findings into a final response
   │    - Produce final output → should_continue routes to END
```

**Key design decisions:**

- **Recursive, not iterative.** Each child is a full graph with its own checkpointed state and tool access controls.
- **LLM controls parallelism.** Multiple `spawn_agent` calls in one response default to `parallel: true` and fan out via `asyncio.gather`. Setting `parallel: false` runs them sequentially.
- **Children can spawn children.** If the child's blueprint includes `spawn_agent` in `tools_allowed`, it can recursively call `run_agent_graph()` — creating grandchild graphs. Budget halving + depth cap prevent infinite recursion.
- **All graphs share infrastructure.** Every graph in the tree shares the same `CostTracker` and `Spawner`. Token usage tracked per-agent via agent IDs.
