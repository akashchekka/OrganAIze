# Evolve — Self-Evolving Agentic AI

> An architecture where agents evolve given a problem, just like humans evolve.

---

## High-Level Architecture

```
                          ┌─────────────────────┐
                          │     User Request    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Genesis Agent     │
                          │   (The Orchestrator)│
                          │   depth=0, budget=N │
                          └──┬──────┬──────┬────┘
                             │      │      │
                    ┌────────▼┐  ┌──▼────┐ ┌▼────────┐
                    │ Agent A │  │Agent B│ │ Agent C  │
                    │ depth=1 │  │depth=1│ │ depth=1  │
                    └──┬───┬──┘  └───────┘ └──────────┘
                       │   │
                 ┌─────▼-┐ ┌▼─────┐
                 │Sub A1 │ │Sub A2│
                 │depth=2│ │depth=2│
                 └───────┘ └──────┘

        ┌─────────────────────────────────────────────────────────┐
        │                Shared Infrastructure                      │
        │                                                           │
        │  ┌──────────────┐  ┌────────────────────┐  ┌───────────┐ │
        │  │ MongoDB       │  │ Redis              │  │ Object    │ │
        │  │               │  │                    │  │ Storage   │ │
        │  │ • agents      │  │ • blackboard (KV)  │  │ (S3/Blob/ │ │
        │  │   (registry)  │  │ • streams (bus)    │  │  MinIO)   │ │
        │  │ • events      │  │ • pub/sub          │  │           │ │
        │  │   (audit log) │  │ • locks (Redlock)  │  │ • images  │ │
        │  │ • tokens      │  │ • heartbeats (TTL) │  │ • code    │ │
        │  │   (usage)     │  │                    │  │ • reports │ │
        │  └──────────────┘  └────────────────────┘  └───────────┘ │
        └─────────────────────────────────────────────────────────────┘
```

### Infrastructure — Clear Boundaries

| Data | Where | Why |
|---|---|---|
| Agent blueprints, lineage, status | MongoDB `agents` | Rich queries ("find hibernated agents with vision expertise") |
| Token usage (own + subtree) | MongoDB `agents` | Lives with the agent record, queried at reporting time |
| Event log (thoughts, tool calls, spawns) | MongoDB `events` | Append-only, time-range queries, aggregation pipelines |
| Blackboard (shared working memory) | Redis KV | Sub-millisecond reads, key TTL, high-frequency access |
| Agent heartbeats / TTL | Redis KV with expiry | Precise per-second TTL enforcement |
| Task dispatch & coordination | Redis Streams | Consumer groups, ordered delivery, agents react without polling |
| "Result ready" notifications | Redis Pub/Sub | Push-based — agents react instantly when a dependency posts results |
| Spawn budget locking | Redis Redlock | Prevents race conditions when two agents try to use the last spawn slot simultaneously |
| Large artifacts (images, code, datasets) | Object Storage (S3/Blob/MinIO) | Unbounded size, cheap, durable. MongoDB stores only a reference URL. |

---

## 1. Spawning Trigger — The Philosophy of Self-Division

An agent shouldn't spawn because it *can*, but because it *must*. Three philosophies govern this:

### Philosophy A: Cognitive Load Detection (Primary)

The agent scores its own task on three axes before acting:

```python
class SpawnAssessment:
    breadth: float    # How many distinct domains does this touch? (0-1)
    depth: float      # How deep is the expertise needed? (0-1)
    parallelism: float  # Can subtasks run independently? (0-1)

    @property
    def should_spawn(self) -> bool:
        # Spawn when breadth is high (multi-domain) AND parallelism allows it
        # Don't spawn for deep-but-narrow tasks — just think harder
        return (self.breadth > 0.6 and self.parallelism > 0.4) or self.breadth > 0.85
```

The LLM performs this assessment via a structured output call:

```
You are evaluating whether to decompose your current task.
Task: "{task_description}"

Score on three axes (0.0 to 1.0):
- breadth: How many distinct skill domains does this require?
- depth: How specialized is the knowledge needed?
- parallelism: Can subtasks proceed independently?

Also: Can an existing agent (check registry) handle this instead of spawning new?
```

### Philosophy B: Failure-Driven Spawning (Reactive)

If an agent attempts a subtask and fails (bad output, tool error, low confidence), it can spawn a *specialist* for that specific failure:

```python
if attempt.confidence < 0.4 or attempt.error:
    specialist = spawn_agent(
        role="specialist",
        task=attempt.failed_subtask,
        traits={"expertise": detect_needed_expertise(attempt.error)},
        reason=f"I failed at: {attempt.error}. Need a specialist."
    )
```

### Philosophy C: Reuse-First Principle (Anti-Spawn)

Before spawning, **always** check the Agent Registry:

```python
def maybe_spawn(task_spec):
    # 1. Check if a living agent can handle it
    existing = registry.find_agent(
        status="alive",
        capabilities__overlap=task_spec.required_skills
    )
    if existing:
        return assign_task(existing, task_spec)

    # 2. Check if a hibernated agent matches
    hibernated = registry.find_agent(
        status="hibernated",
        capabilities__overlap=task_spec.required_skills
    )
    if hibernated:
        return resurrect_and_assign(hibernated, task_spec)

    # 3. Only then spawn new
    return spawn_new_agent(task_spec)
```

### Spawn Budget (Solves: Infinite Spawning)

Every agent is born with a **spawn budget** inherited from its parent:

```python
MAX_GLOBAL_AGENTS = 20  # Hard ceiling
MAX_DEPTH = 4           # No deeper than 4 levels

def calculate_child_budget(parent):
    return SpawnBudget(
        max_children=max(0, parent.budget.max_children // 2),  # Halves each level
        max_depth=parent.budget.max_depth - 1,
        remaining_global=global_agent_count.remaining()
    )
```

Rules:
- **Depth limit = 4.** No agent at depth 4 can spawn.
- **Budget halving.** A depth-0 agent with budget 8 gives each child budget 4, grandchildren budget 2, etc.
- **Global cap = 20.** Once 20 agents exist, no spawning allowed regardless of budget.
- **Spawn requires justification.** Logged reason string — if post-evaluation shows wasteful spawns, the genesis prompt is tuned.

---

## 2. Agent Trait Generation — The DNA Blueprint

Every agent is defined by an **AgentBlueprint** — a structured document that the parent agent generates and the system interprets:

### The AgentBlueprint Schema

```python
class AgentBlueprint:
    # Identity
    agent_id: str           # UUID, auto-generated
    name: str               # Human-readable: "ResearchAgent-FloorPlanModels"
    lineage: list[str]      # [genesis_id, parent_id, ...] — full ancestry

    # Persona & Behavior
    role: str               # "researcher" | "engineer" | "critic" | "synthesizer" | "pm"
    persona: str            # One-line personality: "Skeptical ML researcher who demands evidence"
    expertise: list[str]    # ["computer_vision", "stable_diffusion", "controlnet"]
    verbosity: str          # "minimal" | "normal" | "detailed"
    risk_tolerance: str     # "conservative" | "moderate" | "aggressive"

    # Capabilities
    tools_allowed: list[str]  # ["web_search", "code_execute", "file_write"]
    tools_denied: list[str]   # ["spawn_agent"] — depth-4 agents get this
    model: str                # "gpt-4o" | "claude-sonnet" | "gpt-4o-mini" (cost control)

    # Memory Scope
    memory_scope: MemoryScope  # See Communication Model section

    # Lifecycle
    ttl_seconds: int          # Time-to-live. 0 = until task complete.
    max_steps: int            # Max LLM calls this agent can make
    priority: str             # "critical" | "normal" | "background"

    # Task
    task: str                 # Clear task description
    success_criteria: str     # How to know the task is done
    output_format: str        # "json" | "markdown" | "code" | "structured_report"
```

### System Prompt Generation

The parent agent doesn't write raw system prompts. It fills in the `AgentBlueprint`, and a **prompt compiler** turns it into a system prompt:

```python
def compile_system_prompt(blueprint: AgentBlueprint) -> str:
    return f"""You are {blueprint.name}, a {blueprint.role}.
Personality: {blueprint.persona}
Expertise: {', '.join(blueprint.expertise)}
Verbosity: {blueprint.verbosity}

YOUR TASK:
{blueprint.task}

SUCCESS CRITERIA:
{blueprint.success_criteria}

OUTPUT FORMAT:
{blueprint.output_format}

RULES:
- You have {blueprint.max_steps} steps maximum.
- Tools available: {blueprint.tools_allowed}
- Tools denied: {blueprint.tools_denied}
- Risk tolerance: {blueprint.risk_tolerance}
- You {'CAN' if 'spawn_agent' in blueprint.tools_allowed else 'CANNOT'} spawn sub-agents.
{'- Spawn budget: ' + str(blueprint.spawn_budget) if 'spawn_agent' in blueprint.tools_allowed else ''}

COMMUNICATION:
- Write findings to your blackboard namespace: {blueprint.memory_scope.write_ns}
- You can read from: {blueprint.memory_scope.read_ns}
- Mark keys as shared by prefixing with "shared:"
"""
```

### Model Selection by Role (Solves: Cost Explosion)

Not every agent needs the most expensive model:

| Role | Default Model | Rationale |
|---|---|---|
| Genesis / Orchestrator | gpt-4o / claude-opus | Needs best reasoning for decomposition |
| Researcher | gpt-4o | Needs good synthesis of search results |
| Engineer | claude-sonnet / gpt-4o | Good at code generation |
| Critic / QA | gpt-4o-mini | Checking is cheaper than creating |
| Summarizer | gpt-4o-mini | Compression task, lightweight |
| PM / Tracker | gpt-4o-mini | Status tracking, minimal reasoning |

The parent agent can override this, but defaults keep costs bounded.

---

## 3. Communication Model — Shared Blackboard + Private Memory

### Architecture: Namespaced Blackboard on Redis

```
Redis Key Structure:
──────────────────────────────────────────────
  blackboard:{session_id}:shared:*         ← All agents can read
  blackboard:{session_id}:agent:{agent_id}:* ← Only this agent (and parent) can read
  blackboard:{session_id}:team:{team_tag}:*  ← Agents with same team_tag can read
──────────────────────────────────────────────
```

### MemoryScope Definition

```python
class MemoryScope:
    write_ns: str          # "agent:{self.id}" — always writes to own namespace
    read_ns: list[str]     # ["shared", "agent:{self.id}", "team:research"]
    share_policy: str      # "manual" | "auto_conclusions" | "full_transparency"
```

**Share policies:**
- `manual` — Agent explicitly calls `publish_to_shared(key, value)` when it decides something is worth sharing.
- `auto_conclusions` — Agent's final output is auto-copied to `shared:` namespace. Intermediate thoughts stay private.
- `full_transparency` — Everything the agent writes is mirrored to `shared:`. Used for PM/tracker agents.

### How Agents Communicate

```python
class BlackboardClient:
    def __init__(self, agent_id, session_id, memory_scope, redis):
        self.agent_id = agent_id
        self.session_id = session_id
        self.scope = memory_scope
        self.redis = redis
        self._stream_key = f"stream:{session_id}"  # Redis Stream for this session

    # ── Key-Value (Blackboard) ──────────────────────────────────

    def write(self, key: str, value: any):
        """Write to agent's private namespace."""
        full_key = f"blackboard:{self.session_id}:agent:{self.agent_id}:{key}"
        self.redis.set(full_key, json.dumps(value))
        self._log("write", key, value)

    def share(self, key: str, value: any):
        """Publish to shared namespace + notify listeners via pub/sub."""
        full_key = f"blackboard:{self.session_id}:shared:{key}"
        self.redis.set(full_key, json.dumps(value))
        # Push notification so waiting agents react immediately
        self.redis.publish(
            f"notify:{self.session_id}",
            json.dumps({"agent": self.agent_id, "key": key, "event": "shared"})
        )
        self._log("share", key, value)

    def read_shared(self, pattern: str = "*") -> dict:
        """Read all shared findings."""
        keys = self.redis.keys(f"blackboard:{self.session_id}:shared:{pattern}")
        return {k: json.loads(self.redis.get(k)) for k in keys}

    def read_mine(self, pattern: str = "*") -> dict:
        """Read own private memory."""
        keys = self.redis.keys(f"blackboard:{self.session_id}:agent:{self.agent_id}:{pattern}")
        return {k: json.loads(self.redis.get(k)) for k in keys}

    # ── Pub/Sub (Push Notifications) ────────────────────────────

    def wait_for_key(self, key: str, timeout: int = 60) -> any:
        """Block until a specific shared key appears. No polling."""
        # Check if already available
        existing = self.redis.get(f"blackboard:{self.session_id}:shared:{key}")
        if existing:
            return json.loads(existing)

        # Subscribe and wait for notification
        pubsub = self.redis.pubsub()
        pubsub.subscribe(f"notify:{self.session_id}")
        deadline = time.time() + timeout
        for message in pubsub.listen():
            if time.time() > deadline:
                return None  # Timed out — agent reports partial results to parent
            if message["type"] == "message":
                data = json.loads(message["data"])
                if data["key"] == key:
                    pubsub.unsubscribe()
                    return json.loads(
                        self.redis.get(f"blackboard:{self.session_id}:shared:{key}")
                    )

    # ── Streams (Task Dispatch) ─────────────────────────────────

    def dispatch_task(self, target_agent_id: str, task: dict):
        """Send a task to a specific agent via Redis Stream."""
        self.redis.xadd(self._stream_key, {
            "from": self.agent_id,
            "to": target_agent_id,
            "type": "task",
            "payload": json.dumps(task)
        })

    def consume_tasks(self, consumer_group: str = None) -> list[dict]:
        """Read tasks dispatched to this agent from the stream."""
        group = consumer_group or f"group:{self.agent_id}"
        # Create consumer group if not exists
        try:
            self.redis.xgroup_create(self._stream_key, group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists
        messages = self.redis.xreadgroup(group, self.agent_id, {self._stream_key: ">"}, count=10)
        return [
            json.loads(msg["payload"])
            for stream_msgs in messages
            for _, msg in stream_msgs[1]
            if msg.get("to") == self.agent_id
        ]

    # ── Artifact Storage ────────────────────────────────────────

    def store_artifact(self, filename: str, data: bytes, object_store) -> str:
        """Store large output (images, code, datasets) in object storage.
        Returns a reference URL stored in the blackboard."""
        artifact_path = f"{self.session_id}/{self.agent_id}/{filename}"
        url = object_store.put(artifact_path, data)
        # Store reference in private namespace
        self.write(f"artifact:{filename}", {"url": url, "size": len(data)})
        return url
```

### Agent Registry in MongoDB

```javascript
// agents collection
{
  "_id": "agent_uuid_123",
  "session_id": "session_abc",
  "name": "ResearchAgent-FloorPlanModels",
  "lineage": ["genesis_001", "agent_uuid_123"],
  "parent_id": "genesis_001",
  "depth": 1,

  // Blueprint (the DNA — enough to resurrect this agent)
  "blueprint": {
    "role": "researcher",
    "persona": "Thorough ML researcher who cites sources",
    "expertise": ["computer_vision", "generative_models"],
    "tools_allowed": ["web_search", "read_paper"],
    "tools_denied": ["spawn_agent"],
    "model": "gpt-4o",
    "system_prompt_compiled": "...",  // Cached compiled prompt
    "task": "Find best models for floor-plan-to-image generation",
    "success_criteria": "Ranked list of 3+ models with pros/cons",
    "output_format": "structured_report"
  },

  // Lifecycle
  "status": "alive",          // alive | completed | hibernated | dead
  "created_at": "2026-03-17T10:00:00Z",
  "ttl_seconds": 300,
  "expires_at": "2026-03-17T10:05:00Z",
  "steps_taken": 4,
  "max_steps": 20,

  // Relationships
  "children": ["agent_uuid_456", "agent_uuid_789"],
  "spawn_budget_remaining": 0,

  // Token Usage (own) — this agent's direct LLM calls only
  "token_usage": {
    "input_tokens": 4200,
    "output_tokens": 1800,
    "total_tokens": 6000,
    "llm_calls": 3,
    "cost_usd": 0.018
  },

  // Token Usage (subtree) — this agent + ALL descendants, rolled up
  "subtree_token_usage": {
    "input_tokens": 18400,
    "output_tokens": 9200,
    "total_tokens": 27600,
    "llm_calls": 14,
    "cost_usd": 0.073,
    "agent_count": 3          // self + 2 children
  },

  // Results
  "final_output": null,       // Populated on completion
  "exit_reason": null         // "completed" | "ttl_expired" | "budget_exhausted" | "killed_by_parent"
}
```

### Context Aggregation (Solves: Context Window Blow-up)

When the Genesis agent collects results from sub-agents, it doesn't dump raw outputs into its context. Instead:

```python
class ResultAggregator:
    def aggregate(self, child_results: list[AgentResult]) -> str:
        summaries = []
        for result in child_results:
            # Each agent's output was already in output_format
            # Compress further if needed
            if len(result.output) > 2000:
                summary = llm_mini.summarize(result.output, max_tokens=500)
            else:
                summary = result.output
            summaries.append(f"[{result.agent_name}]: {summary}")

        return "\n---\n".join(summaries)
```

Rules:
- Sub-agent outputs are capped at a token budget per agent.
- If an agent's output exceeds the cap, a cheap model summarizes it.
- Full outputs remain on the blackboard — the parent can drill down if the summary is insufficient.

---

## 4. Lifecycle — Birth, Sleep, Resurrection, Death

### State Machine

```
                  spawn()
    ┌─────────┐ ──────────► ┌─────────┐
    │  VOID   │              │  ALIVE  │ ◄──── resurrect()
    └─────────┘              └────┬────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │               │
              task_done()    ttl_expired()    budget_exhausted()
                    │             │               │
                    ▼             ▼               ▼
             ┌──────────┐  ┌──────────┐   ┌──────────┐
             │COMPLETED  │  │HIBERNATED│   │  DEAD    │
             └──────────┘  └─────┬────┘   └──────────┘
                                 │
                           resurrect()
                                 │
                                 ▼
                           ┌──────────┐
                           │  ALIVE   │
                           └──────────┘
```

### Key Lifecycle Rules

| Transition | Trigger | What Happens |
|---|---|---|
| → ALIVE | `spawn()` or `resurrect()` | Agent starts executing. TTL countdown begins. |
| ALIVE → COMPLETED | Task done, success criteria met | Final output saved. Blackboard entries preserved. Blueprint kept in registry for potential reuse. |
| ALIVE → HIBERNATED | TTL expires BUT task was partially done | Agent state (last step, partial results) saved to registry. Can be woken up. Blackboard entries preserved. |
| ALIVE → DEAD | Budget exhausted OR killed by parent OR max_steps hit with no progress | Blueprint saved for forensics. Blackboard entries cleaned after session. |
| HIBERNATED → ALIVE | Parent or new agent calls `resurrect(agent_id)` | Agent resumes from saved state. New TTL assigned. |

### TTL & Heartbeat

```python
class AgentLifecycle:
    def __init__(self, agent_id, ttl_seconds, registry, redis):
        self.agent_id = agent_id
        self.ttl = ttl_seconds
        # Set a Redis key with TTL for automatic expiration detection
        redis.setex(f"heartbeat:{agent_id}", ttl_seconds, "alive")

    def step(self):
        """Called after each agent action. Checks if still alive."""
        if not self.redis.exists(f"heartbeat:{self.agent_id}"):
            self.hibernate()  # TTL expired between steps
            raise AgentExpired()

    def extend_ttl(self, extra_seconds):
        """Agent can request more time if making progress."""
        remaining = self.redis.ttl(f"heartbeat:{self.agent_id}")
        self.redis.expire(f"heartbeat:{self.agent_id}", remaining + extra_seconds)
        self._log("ttl_extended", extra_seconds)

    def hibernate(self):
        """Save state and go to sleep."""
        state = self.capture_state()
        self.registry.update(self.agent_id, status="hibernated", saved_state=state)
```

### Resurrection Protocol

```python
def resurrect(agent_id, new_task=None):
    record = registry.find_by_id(agent_id)
    assert record["status"] == "hibernated"

    blueprint = record["blueprint"]
    saved_state = record.get("saved_state", {})

    # Rehydrate the agent with its original DNA + saved progress
    agent = Agent(
        blueprint=blueprint,
        initial_context=f"""You were previously working on this task and were paused.
Your progress so far: {saved_state.get('partial_results', 'None')}
Last step completed: {saved_state.get('last_step', 'None')}
{'New additional task: ' + new_task if new_task else 'Resume where you left off.'}"""
    )

    registry.update(agent_id, status="alive", ttl_seconds=blueprint.ttl_seconds)
    return agent
```

---

## 5. Logging — The Immutable Event Journal

Every agent action produces an **event** appended to an immutable log. Nothing is ever deleted from this log.

### Event Schema

```python
class AgentEvent:
    event_id: str           # UUID
    timestamp: datetime     # ISO 8601
    session_id: str
    agent_id: str
    agent_name: str
    depth: int
    event_type: str         # See event types below
    payload: dict           # Event-specific data
    parent_event_id: str    # For causal chain tracing (optional)
```

### Event Types

| Event Type | When | Payload |
|---|---|---|
| `agent.spawned` | Agent created | `{blueprint, parent_id, spawn_reason}` |
| `agent.thought` | LLM internal reasoning | `{thought_text, step_number}` |
| `agent.tool_call` | Tool invoked | `{tool_name, args, result, latency_ms}` |
| `agent.tool_error` | Tool failed | `{tool_name, args, error}` |
| `agent.spawn_decision` | Decided to spawn (or not) | `{assessment: SpawnAssessment, decision, reason}` |
| `agent.blackboard_write` | Wrote to blackboard | `{namespace, key, value_preview}` |
| `agent.blackboard_read` | Read from blackboard | `{namespace, pattern, keys_returned}` |
| `agent.output` | Produced final output | `{output, meets_success_criteria}` |
| `agent.hibernated` | TTL expired, going to sleep | `{saved_state_summary}` |
| `agent.resurrected` | Woken from hibernation | `{new_task, previous_state}` |
| `agent.died` | Terminated | `{exit_reason, steps_taken}` |
| `agent.cost` | LLM call made | `{model, input_tokens, output_tokens, cost_usd}` |

### Storage & Querying

```python
# MongoDB collection: events (capped or TTL-indexed for cleanup)
# Index on: session_id, agent_id, event_type, timestamp

class EventLogger:
    def __init__(self, db):
        self.collection = db["events"]

    def log(self, event: AgentEvent):
        self.collection.insert_one(asdict(event))

    def get_agent_trace(self, agent_id) -> list[AgentEvent]:
        """Full chronological trace of one agent's life."""
        return list(self.collection.find(
            {"agent_id": agent_id}
        ).sort("timestamp", 1))

    def get_session_tree(self, session_id) -> list[AgentEvent]:
        """Everything that happened in a session — for post-mortem."""
        return list(self.collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1))

    def get_spawn_tree(self, session_id) -> dict:
        """Reconstruct the agent hierarchy for visualization."""
        spawns = self.collection.find({
            "session_id": session_id,
            "event_type": "agent.spawned"
        })
        # Build tree from lineage data
        ...
```

### Post-Evaluation Dashboard (What You Can Answer)

From this log, you can answer:
- **How many agents were spawned?** Count `agent.spawned` events.
- **Was spawning efficient?** Compare agents that produced useful output vs. those that died without output.
- **What was the total cost?** Sum all `agent.cost` events.
- **Where did time go?** Timeline visualization from timestamps.
- **Did any agent loop?** Detect repeated `agent.thought` patterns.
- **What was the critical path?** Trace from final output back through `parent_event_id` chains.

---

## 6. Addressing the Hard Problems

### Infinite Spawning → Budget + Depth + Global Cap

Three independent layers of defense. No single agent can circumvent all three.

```
Spawn request arrives at spawner._create_agent()
    │
    ├── 1. DEPTH check: max_depth_remaining > 0?   → NO → denied
    ├── 2. BREADTH check: max_children > 0?         → NO → denied
    ├── 3. GLOBAL check: alive_count < 20?           → NO → denied
    ├── 4. Redlock acquired?                         → NO → denied
    └── All pass → spawn allowed
```

**Layer 1: Depth Limit (vertical control)**

Prevents infinite nesting. Each child's `max_depth_remaining` = parent's - 1.

```python
# SpawnBudget.can_spawn (blueprint.py)
@property
def can_spawn(self) -> bool:
    return (
        self.max_children > 0
        and self.max_depth_remaining > 0   # ← depth gate
        and self.remaining_global > 0
    )
```

At depth = `MAX_DEPTH` (4), agents get `spawn_agent` added to `tools_denied` — the LLM never even sees the option:

```python
# spawner.py
can_spawn = spec.get("needs_spawn", False) and child_depth < MAX_DEPTH
tools_denied = [] if can_spawn else ["spawn_agent"]
```

**Layer 2: Budget Halving (horizontal control)**

Prevents sibling explosion. Each level gets half the parent's budget:

| Depth | max_children | Calculation |
|---|---|---|
| 0 (Genesis) | 8 | `GENESIS_SPAWN_BUDGET` |
| 1 | 4 | `8 // 2` |
| 2 | 2 | `4 // 2` |
| 3 | 1 | `2 // 2` |
| 4 | 0 | Cannot spawn |

Each successful spawn decrements the parent's budget:

```python
# spawner.py — after creating a child
parent.spawn_budget.max_children -= 1
await self.registry.update(parent.agent_id, spawn_budget_remaining=parent.spawn_budget.max_children)
```

**Layer 3: Global Cap (absolute control)**

Regardless of individual budgets, once `MAX_GLOBAL_AGENTS` (20) exist in the session, all spawning stops:

```python
alive_count = await self.registry.count_alive(self.session_id)
if alive_count >= MAX_GLOBAL_AGENTS:
    # spawn denied — logged as agent.spawn_decision event
```

**All three are enforced under a Redis distributed lock** (`Redlock`) to prevent race conditions where two agents try to spend the last budget slot simultaneously.

### Context Window Blow-up → Hierarchical Summarization

```
Agent outputs are not dumped raw into parent context.
Each child output → summarized if > 2000 chars.
Full data stays on blackboard for drill-down.
Parent can ask: "Agent X, expand on point 3" via targeted read.
```

### Tool Safety → Allowlist per Agent

```python
# Tools are allowlisted, not blocklisted.
# Each agent only gets the tools its blueprint specifies.
# Dangerous tools (file_delete, shell_exec) require:
#   1. Blueprint explicitly lists them
#   2. Agent depth <= 1 (only direct children of genesis)
#   3. Confirmation event logged before execution

RESTRICTED_TOOLS = {"shell_exec", "file_delete", "network_request_external"}

def validate_tool_access(agent, tool_name):
    if tool_name in RESTRICTED_TOOLS:
        if agent.depth > 1:
            raise ToolDenied(f"{tool_name} restricted to depth <= 1")
        log_event("agent.restricted_tool_use", {
            "tool": tool_name, "agent": agent.id, "approved": True
        })
```

### Cost Explosion → Budget Tracking + Model Tiering + Per-Agent Token Accounting

```python
class TokenUsage:
    """Tracks token consumption for a single agent."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def record(self, model: str, inp: int, out: int):
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += (inp + out)
        self.llm_calls += 1
        self.cost_usd += calculate_cost(model, inp, out)


class CostTracker:
    """
    Session-wide cost tracker. Also maintains per-agent token ledgers
    and rolls up subtree totals to parent agents in the registry.
    """
    def __init__(self, session_budget_usd: float = 1.00, registry=None):
        self.budget = session_budget_usd
        self.spent = 0.0
        self.registry = registry
        self.agent_usage: dict[str, TokenUsage] = {}  # agent_id → TokenUsage

    def charge(self, agent_id: str, model: str, input_tokens: int, output_tokens: int):
        cost = calculate_cost(model, input_tokens, output_tokens)
        self.spent += cost

        # Track per-agent tokens
        if agent_id not in self.agent_usage:
            self.agent_usage[agent_id] = TokenUsage()
        self.agent_usage[agent_id].record(model, input_tokens, output_tokens)

        # Persist to agent registry (MongoDB) after every call
        self.registry.update(agent_id, token_usage=asdict(self.agent_usage[agent_id]))

        # Session-level budget alerts
        if self.spent > self.budget * 0.8:
            notify_genesis("80% budget consumed")
        if self.spent >= self.budget:
            raise BudgetExhausted()
        return cost

    def rollup_subtree(self, agent_id: str):
        """
        Called when an agent completes. Walks up the lineage to update
        every ancestor's subtree_token_usage with this agent's own tokens.
        """
        agent = self.registry.find_by_id(agent_id)
        own = self.agent_usage.get(agent_id, TokenUsage())

        # Update own subtree (self + children already rolled up)
        children_subtrees = [
            self.registry.find_by_id(cid).get("subtree_token_usage", {})
            for cid in agent.get("children", [])
        ]
        subtree = TokenUsage(
            input_tokens=own.input_tokens + sum(c.get("input_tokens", 0) for c in children_subtrees),
            output_tokens=own.output_tokens + sum(c.get("output_tokens", 0) for c in children_subtrees),
            total_tokens=own.total_tokens + sum(c.get("total_tokens", 0) for c in children_subtrees),
            llm_calls=own.llm_calls + sum(c.get("llm_calls", 0) for c in children_subtrees),
            cost_usd=own.cost_usd + sum(c.get("cost_usd", 0) for c in children_subtrees),
        )
        agent_count = 1 + sum(c.get("agent_count", 1) for c in children_subtrees)

        self.registry.update(agent_id, subtree_token_usage={
            **asdict(subtree), "agent_count": agent_count
        })

    def get_session_summary(self) -> dict:
        """Total token usage across all agents in the session."""
        return {
            "total_input_tokens": sum(u.input_tokens for u in self.agent_usage.values()),
            "total_output_tokens": sum(u.output_tokens for u in self.agent_usage.values()),
            "total_tokens": sum(u.total_tokens for u in self.agent_usage.values()),
            "total_llm_calls": sum(u.llm_calls for u in self.agent_usage.values()),
            "total_cost_usd": self.spent,
            "per_agent": {
                aid: asdict(usage) for aid, usage in self.agent_usage.items()
            }
        }

# Model tiering: cheap models for cheap tasks (see section 2)
# Parallel execution capped: max 3 concurrent LLM calls
```

**How token data flows:**

```
Agent makes LLM call
    │
    ▼
CostTracker.charge(agent_id, model, in, out)
    ├── Updates in-memory TokenUsage for this agent
    ├── Persists token_usage to MongoDB (agent's own row)
    └── Checks session budget
    │
Agent completes
    │
    ▼
CostTracker.rollup_subtree(agent_id)
    ├── Sums own tokens + all children's subtree_token_usage
    ├── Writes subtree_token_usage to MongoDB
    └── Parent can now see: "my subtree used X tokens total"
    │
Genesis completes
    │
    ▼
CostTracker.get_session_summary()
    └── Returns full breakdown: per-agent + session totals
```

### Deadlocks → No Circular Dependencies by Design

```
Rule: Information flows DOWN (parent → child) and UP (child → parent).
Agents at the same depth NEVER wait on each other directly.
They communicate via blackboard + pub/sub (push-based, not polling).
If Agent A needs Agent B's output:
  → A calls wait_for_key("research:top_models", timeout=60)
  → Redis pub/sub pushes notification the instant B shares the key
  → No CPU-wasting polls

wait_timeout = 60 seconds
on_timeout → report partial results to parent, let parent decide
```

### Spawn Race Conditions → Redis Distributed Locks

```python
def safe_spawn(parent, task_spec):
    """Acquire a Redlock before spending spawn budget.
    Prevents two concurrent agents from using the last slot."""
    lock = redis.lock(f"spawn_lock:{parent.session_id}", timeout=5)
    if lock.acquire(blocking_timeout=3):
        try:
            remaining = global_agent_count.remaining()
            if remaining <= 0:
                raise SpawnDenied("Global agent cap reached")
            return spawn_new_agent(task_spec)
        finally:
            lock.release()
    else:
        raise SpawnDenied("Could not acquire spawn lock")
```

---

## 7. Putting It All Together — Full Flow

```
1. User submits: "Build an app that converts floor plans into realistic images"

2. Genesis Agent receives task (LangGraph StateGraph)
   ├── reason_node: Runs SpawnAssessment: breadth=0.9, depth=0.7, parallelism=0.8
   ├── should_continue → tools (spawn_agent tool calls)
   ├── tool_node: Checks registry → no existing agents → spawn fresh
   └── Spawns child StateGraphs via run_agent_graph():
       ├── ResearchAgent graph (model=gpt-4o, tools=[web_search], ttl=300s)
       ├── ArchitectAgent graph (model=claude-sonnet, tools=[code_exec], ttl=300s)
       └── PMAgent graph (model=gpt-4o-mini, tools=[], share_policy=full_transparency)

3. Child agent graphs run (each with own checkpointed StateGraph)
   ├── ResearchAgent:
   │   ├── Searches web for floor-plan-to-image models
   │   ├── Writes to private: "controlnet_notes", "sd_comparison"
   │   ├── Shares to blackboard: "research:top_models" → ranked list + pub/sub notify
   │   └── Completes → status=COMPLETED
   │
   ├── ArchitectAgent:
   │   ├── wait_for_key("research:top_models") → wakes instantly via pub/sub
   │   ├── Runs SpawnAssessment on "build full app": breadth=0.7
   │   ├── Spawns: FrontendAgent, BackendAgent (depth=2, budget=1 each)
   │   │   ├── FrontendAgent builds React UI → COMPLETED
   │   │   └── BackendAgent builds FastAPI + ControlNet pipeline → COMPLETED
   │   │       └── Stores generated sample images → Object Storage (S3/Blob)
   │   ├── Collects child outputs, integrates
   │   └── Shares: "architecture:final_design" → complete app structure
   │
   └── PMAgent:
       ├── Monitors all shared: entries
       ├── Detects: BackendAgent taking long → logs warning
       └── Reports status to Genesis when all agents complete

4. Genesis Agent graph continues (reason_node receives tool results):
   ├── Reads aggregated outputs (summarized if > 3000 chars)
   ├── reason_node synthesizes final response
   ├── should_continue → END (no more tool calls)
   ├── All agents → COMPLETED or DEAD
   ├── Checkpointed state available for post-mortem replay
   └── Session events archived for post-evaluation

5. Post-evaluation:
   ├── 6 agents spawned, 6 produced useful output → 100% efficiency
   ├── Total cost: $0.23
   ├── Total time: 45 seconds (parallel execution)
   └── Spawn tree visualization generated from event log
```

---

## 8. Recommended Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Agent Runtime** | LangGraph `StateGraph` | Each agent is a `reason → tools → reason` graph. Dynamic node creation for spawned agents. Built-in checkpointing for hibernate/resume. |
| **LLM Calls** | LiteLLM wrapper | Unified API across OpenAI, Anthropic, local models — easy model tiering |
| **State & Checkpointing** | LangGraph `MemorySaver` (dev) / PostgreSQL checkpointer (prod) | Automatic mid-step state persistence. Enables crash recovery and hibernate/resurrect without manual serialization. |
| **MongoDB** | Agent registry + event log + token tracking | Flexible schema for blueprints, rich queries, aggregation pipelines for analytics |
| **Redis** | Blackboard (KV) + Streams (bus) + Pub/Sub (notifications) + Redlock (concurrency) + Heartbeats (TTL) | Sub-ms reads, push-based coordination, precise TTL, distributed locking |
| **Object Storage** | S3 / Azure Blob / MinIO | Large artifacts (images, code, datasets). MongoDB stores only reference URLs. |
| **Orchestration** | asyncio + semaphore | Parallel agent execution with concurrency cap |
| **Dashboard** | Streamlit or Grafana | Visualize spawn trees, cost, timelines from event log |

---

## 9. Project Structure

```
Evolve/
├── core/
│   ├── genesis.py          # Genesis Agent — top-level orchestrator (LangGraph)
│   ├── agent.py            # LangGraph StateGraph: reason → tools → reason loop
│   │                        #   AgentState, reason_node, tool_node, should_continue
│   │                        #   build_agent_graph(), run_agent_graph()
│   ├── blueprint.py        # AgentBlueprint schema + prompt compiler
│   ├── spawner.py          # Spawn logic: assessment, budget, reuse-first
│   ├── lifecycle.py        # TTL heartbeat, hibernate, resurrect, death
│   └── cost_tracker.py     # Budget tracking, model cost tables, token rollup
│
├── memory/
│   ├── blackboard.py       # Redis KV + Pub/Sub + Streams + cleanup
│   ├── locks.py            # Redis Redlock — spawn budget concurrency control
│   ├── registry.py         # MongoDB agent registry (DNA storage)
│   ├── artifacts.py        # Object storage client (S3/Blob/MinIO)
│   └── scoping.py          # MemoryScope, namespace logic
│
├── logging/
│   ├── event_logger.py     # Immutable event journal (MongoDB)
│   ├── event_types.py      # Event type constants
│   └── trace.py            # Spawn tree + critical path reconstruction
│
├── prompts/
│   ├── genesis.py          # GENESIS_PERSONA
│   ├── spawner.py          # ASSESSMENT_PROMPT, DECOMPOSITION_PROMPT
│   ├── agent.py            # AGENT_SYSTEM_PROMPT_TEMPLATE, SPAWN_ALLOWED/DENIED
│   └── lifecycle.py        # RESURRECTION_CONTEXT_TEMPLATE
│
├── tools/
│   └── tool_registry.py    # Tool definitions, access control, OpenAI-format specs
│
├── custom_loop/            # Legacy: original while-loop agent runtime (preserved)
│   ├── agent.py            # Custom while-loop Agent class
│   └── genesis.py          # Custom-loop GenesisAgent
│
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   ├── spawn_tree.py       # Tree visualization
│   └── cost_report.py      # Cost analysis views
│
├── config.py               # Global caps, defaults, model pricing
├── main.py                 # CLI entry point
├── requirements.txt
└── DESIGN.md               # This document
```

### LangGraph Agent Architecture

Each agent (including Genesis) runs as a LangGraph `StateGraph`:

```
┌──────────────────────────────────────────────────┐
│              Agent StateGraph                        │
│                                                      │
│  ┌────────┐    should_continue    ┌────────┐          │
│  │ reason ├───► has tool calls? ──►│ tools  ├────┐     │
│  │ (LLM)  │    │                  │ (exec) │    │     │
│  └────┬───┘    │                  └────────┘    │     │
│       ▲        │ no tool calls                 │     │
│       └────────┴───────────────────────────┘     │
│                │                               │     │
│                ▼                               │     │
│           ┌───────┐                           │     │
│           │  END  │                           │     │
│           └───────┘                           │     │
│                                               │     │
│  When tool = spawn_agent:                      │     │
│    tool_node creates a NEW StateGraph           │     │
│    for the child agent and runs it to            │     │
│    completion. Output returned as ToolMessage.   │     │
│                                               │     │
│  Checkpointer: MemorySaver (thread_id=agent_id) │     │
│    → Enables hibernate by saving graph state     │     │
│    → Resume by re-invoking with same thread_id   │     │
└──────────────────────────────────────────────────┘
```

**Why LangGraph over a custom while-loop:**

| Concern | Custom Loop | LangGraph |
|---|---|---|
| **Checkpointing** | Manual `saved_state` dict | Automatic — every node transition is checkpointed |
| **Crash recovery** | Lost — entire session gone | Resume from last checkpoint |
| **Parallel spawning** | Sequential `await child.run()` | Can fan-out multiple child graphs concurrently |
| **Streaming** | Not supported | Built-in token and event streaming |
| **Human-in-the-loop** | Not supported | Pause graph at any node for approval |
| **Visualization** | Manual | LangGraph Studio visualizes live graph execution |

### Dynamic Spawning — How It Works in LangGraph

The graph topology is **structurally static** (`reason → tools → reason`) but **dynamically recursive** — the `tool_node` can create entirely new agent graphs at runtime.

**Step-by-step flow:**

```
1. reason_node (Parent)
   │  LLM decides: "I need a specialist for computer vision research"
   │  Emits AIMessage with tool_call: spawn_agent({
   │      name: "ResearchAgent-CV",
   │      role: "researcher",
   │      task: "Find best floor-plan-to-image models",
   │      ...
   │  })
   │
   ▼
2. should_continue
   │  Sees tool_calls on AIMessage → routes to "tools"
   │
   ▼
3. tool_node (Parent)
   │  Iterates tool calls, hits "spawn_agent"
   │  Calls _handle_spawn(args, parent_blueprint, spawner, deps)
   │
   ▼
4. _handle_spawn()
   │  ├── spawner.maybe_spawn() →
   │  │     ├── Reuse-first: checks registry for alive/hibernated agents
   │  │     ├── Acquires Redlock (prevents race conditions)
   │  │     ├── Checks depth limit, budget, global cap
   │  │     ├── Creates child AgentBlueprint
   │  │     └── Registers in MongoDB, logs spawn event
   │  │
   │  └── run_agent_graph(child_blueprint, ...) →
   │        Creates a BRAND NEW StateGraph for the child:
   │
   │        ┌──────────────────────────────────┐
   │        │     Child Agent StateGraph        │
   │        │                                   │
   │        │  reason ──► tools ──► reason ──►  │
   │        │     │                    │        │
   │        │     └──► END             │        │
   │        │                          │        │
   │        │  (Child can also call     │        │
   │        │   spawn_agent → creating  │        │
   │        │   grandchild graphs!)     │        │
   │        │                           │        │
   │        │  Checkpointer: MemorySaver│        │
   │        │  thread_id = child.agent_id       │
   │        └──────────────────────────────────┘
   │
   │        Child graph runs to completion.
   │        Returns final output string.
   │
   ▼
5. tool_node (Parent) — continued
   │  Receives child output, wraps it in a ToolMessage:
   │  ToolMessage(content='{"tool": "spawn_agent", "status": "completed",
   │      "agent_name": "ResearchAgent-CV", "output": "...findings..."}')
   │
   ▼
6. reason_node (Parent)
   │  LLM sees the ToolMessage with child's output.
   │  Can now:
   │    - Spawn more agents (another spawn_agent tool call → repeat)
   │    - Use the findings to reason further
   │    - Produce final output → should_continue routes to END
```

**Key design decisions:**

- **Recursive, not iterative.** Each child is a full graph, not a function call. This means children get their own checkpointed state, their own TTL, their own tool access controls.
- **Blocking by default.** `_handle_spawn` awaits the child graph. The parent's `tool_node` blocks until the child completes. This keeps the message flow simple — the parent's LLM sees the child output as a tool result.
- **Children can spawn children.** If the child's blueprint includes `spawn_agent` in `tools_allowed`, the child's `tool_node` can recursively call `run_agent_graph()` again — creating grandchild graphs. Budget halving + depth cap prevent infinite recursion.
- **All graphs share infrastructure.** Every graph in the tree shares the same `CostTracker`, `EventLogger`, `Spawner`, and Redis connection. Token usage rolls up through the subtree.

---

## 10. TODOs — What's Done & What's Next

### Completed

- [x] **Core architecture design** — spawning, traits, communication, lifecycle, logging
- [x] **Infrastructure layer** — MongoDB registry, Redis blackboard (KV + pub/sub + streams + locks), object storage
- [x] **AgentBlueprint + prompt compiler** — structured blueprint schema, system prompt generation
- [x] **Cost tracker** — per-agent token accounting, subtree rollup, session budget enforcement
- [x] **Event logger** — immutable event journal with causal chain tracing
- [x] **Spawner** — spawn assessment, reuse-first, budget halving, depth cap, global cap, Redlock
- [x] **Agent lifecycle** — TTL heartbeat, hibernate, resurrect, death state machine
- [x] **Tool registry** — allowlist-based access control, restricted tools by depth
- [x] **LangGraph refactor** — `StateGraph` with `reason → tools → reason` loop, checkpointing via `MemorySaver`
- [x] **Prompt separation** — domain-specific prompt files in `evolve/prompts/`
- [x] **Custom loop preserved** — legacy while-loop approach in `evolve/custom_loop/`

### TODO: Dynamic Skill System

#### Phase 1 — Skill DB + Curated Skills
- [ ] Design MongoDB `skills` collection schema (name, description, keywords, embedding, trust_level, body, assets, stats)
- [ ] Build `SkillRegistry` class — CRUD, keyword search, embedding similarity search
- [ ] Build `Skill` dataclass — parse SKILL.md frontmatter + body + asset catalog
- [ ] Build `SkillMatcher` — three-tier lookup: in-memory cache → MongoDB → fallback
- [ ] Wire skill matching into `Spawner._create_agent()` — enrich blueprint with matched skill
- [ ] Handle skill assets: templates (inject via file_write), scripts (execute via code_execute), examples (append to system prompt)
- [ ] Skill asset sandboxing — scripts run in subprocess with timeout, no network, workspace-restricted
- [ ] Skill trust levels — `verified` | `community` | `discovered`; depth > 1 agents only use `verified`
- [ ] Ship 3-5 curated starter skills (e.g., research, web-app-scaffold, data-analysis, code-review)
- [ ] Add SKILL.md prompts to `evolve/prompts/` for skill-enriched system prompts

#### Phase 2 — Mid-Run Skill Discovery
- [ ] Add `search_skill` tool — agents can search the skill DB during execution (not just at spawn)
- [ ] Add `load_skill` tool — agent can load a skill mid-run, enriching its own system prompt
- [ ] Rate limiting on skill lookups — max 3 skill searches per agent per session
- [ ] Skill loading logged in event journal (`agent.skill_loaded` event type)

#### Phase 3 — Web Search Skill Discovery
- [ ] Web search fallback when skill DB has no match — agent searches web for domain knowledge
- [ ] Skill ingestion pipeline — extract SKILL.md-compatible structure from web results
- [ ] Trust gating — web-discovered skills marked as `discovered`, require validation before use
- [ ] LLM-based skill quality check — evaluate discovered skill for accuracy, relevance, safety
- [ ] Deduplication — check if a semantically similar skill already exists before ingesting

#### Phase 4 — Agent-Authored Skills
- [ ] After successful novel task, agent proposes a new skill based on its experience
- [ ] Skill generation prompt — agent produces SKILL.md with workflow, tools used, key patterns, code templates
- [ ] Auto-saved to DB with `trust_level="discovered"` and `created_by=agent:{id}`
- [ ] Skill rating system — parent agent rates child's skill contribution after use
- [ ] Human review workflow — discovered/agent-authored skills queued for promotion to `community` or `verified`
- [ ] Skill deprecation — flag skills with low success rates or old last-validated timestamps
- [ ] Skill versioning — track version history, allow rollback

### TODO: Production Hardening

- [ ] **LLM resilience** — retry with exponential backoff, model fallback chain (gpt-4o → claude-sonnet → gpt-4o-mini)
- [ ] **Error recovery** — circuit breakers per agent, agent-level isolation (one agent crash doesn't kill session)
- [ ] **Sandboxed code execution** — replace subprocess with container-based sandbox (gVisor/Firecracker) or sandboxed interpreter API
- [ ] **Observability** — OpenTelemetry traces, structured logging, Prometheus metrics for agent count, cost, latency
- [ ] **Scaling** — worker pool (Celery/Dramatiq) or containerized agents for horizontal scaling
- [ ] **Rate limiting** — per-model token/minute tracking matching provider rate limits
- [ ] **Auth & multi-tenancy** — session isolation, API auth, per-tenant budget caps
- [ ] **Real tool implementations** — wire up SerpAPI/Tavily for web_search, PDF parser for read_paper, etc.
- [ ] **LangGraph PostgreSQL checkpointer** — replace MemorySaver with persistent checkpointer for crash recovery across restarts
- [ ] **Parallel child execution** — fan-out multiple child graphs via `asyncio.gather` instead of sequential spawn

### TODO: Dashboard & Observability

- [ ] **Streamlit dashboard** — real-time session monitoring
- [ ] **Spawn tree visualization** — interactive tree from event log spawn events
- [ ] **Cost report views** — per-agent, per-session, per-model cost breakdowns
- [ ] **Token usage heatmap** — which agents consumed the most tokens and why
- [ ] **Replay mode** — step through a completed session event-by-event for post-mortem
- [ ] **Skill usage analytics** — which skills are loaded most, success rates, agent ratings
