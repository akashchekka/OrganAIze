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
@dataclass
class SpawnAssessment:
    breadth: float = 0.0      # How many distinct domains does this touch? (0-1)
    depth: float = 0.0        # How deep is the expertise needed? (0-1)
    parallelism: float = 0.0  # Can subtasks run independently? (0-1)
    reasoning: str = ""       # LLM's reasoning for the scores

    @property
    def should_spawn(self) -> bool:
        # Thresholds from config: SPAWN_BREADTH_THRESHOLD=0.6,
        # SPAWN_PARALLELISM_THRESHOLD=0.4, SPAWN_AUTO_THRESHOLD=0.85
        return (
            (self.breadth > SPAWN_BREADTH_THRESHOLD and self.parallelism > SPAWN_PARALLELISM_THRESHOLD)
            or self.breadth > SPAWN_AUTO_THRESHOLD
        )
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
async def maybe_spawn(
    self,
    parent_blueprint: AgentBlueprint,
    task_spec: dict,
) -> AgentBlueprint | None:
    """Try to reuse an existing agent, resurrect a hibernated one, or spawn new."""
    required_skills = task_spec.get("expertise", [])

    # 1. Check for a living agent that already has the needed skills
    existing = await self.registry.find_agent(
        session_id=self.session_id,
        status="alive",
        capabilities=required_skills,
    )
    if existing and existing["_id"] != parent_blueprint.agent_id:
        return None  # Caller should dispatch_task instead

    # 2. Check for a hibernated agent
    hibernated = await self.registry.find_agent(
        session_id=self.session_id,
        status="hibernated",
        capabilities=required_skills,
    )
    if hibernated:
        return None  # Caller handles lifecycle.resurrect()

    # 3. Spawn new — with lock
    return await self._create_agent(parent_blueprint, task_spec)
```

### Spawn Budget (Solves: Infinite Spawning)

Every agent is born with a **spawn budget** inherited from its parent:

```python
MAX_GLOBAL_AGENTS = int(os.getenv("EVOLVE_MAX_AGENTS", "20"))   # Hard ceiling
MAX_DEPTH = int(os.getenv("EVOLVE_MAX_DEPTH", "4"))             # No deeper than 4 levels
GENESIS_SPAWN_BUDGET = int(os.getenv("EVOLVE_GENESIS_BUDGET", "8"))

# Inside spawner._create_agent():
child_budget = SpawnBudget(
    max_children=max(0, parent.spawn_budget.max_children // 2),
    max_depth_remaining=parent.spawn_budget.max_depth_remaining - 1,
    remaining_global=MAX_GLOBAL_AGENTS - alive_count - 1,
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

The parent agent doesn't write raw system prompts. It fills in the `AgentBlueprint`, and a **prompt compiler** turns it into a system prompt using `AGENT_SYSTEM_PROMPT_TEMPLATE` from `prompts/agent.py`:

```python
def compile_system_prompt(blueprint: AgentBlueprint) -> str:
    """Convert an AgentBlueprint into a system prompt string."""
    if "spawn_agent" in blueprint.tools_allowed:
        spawn_section = SPAWN_ALLOWED_SECTION.format(
            max_children=blueprint.spawn_budget.max_children,
            max_depth_remaining=blueprint.spawn_budget.max_depth_remaining,
        )
    else:
        spawn_section = SPAWN_DENIED_SECTION

    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        name=blueprint.name,
        role=blueprint.role,
        persona=blueprint.persona,
        expertise=', '.join(blueprint.expertise) or 'general',
        verbosity=blueprint.verbosity,
        task=blueprint.task,
        success_criteria=blueprint.success_criteria,
        output_format=blueprint.output_format,
        max_steps=blueprint.max_steps,
        tools_allowed=', '.join(blueprint.tools_allowed) or 'none',
        tools_denied=', '.join(blueprint.tools_denied) or 'none',
        risk_tolerance=blueprint.risk_tolerance,
        spawn_section=spawn_section,
        write_ns=blueprint.memory_scope.write_ns,
        read_ns=', '.join(blueprint.memory_scope.read_ns),
        share_policy=blueprint.memory_scope.share_policy,
    )
```

The template and spawn sections are defined in `prompts/agent.py` as `AGENT_SYSTEM_PROMPT_TEMPLATE`, `SPAWN_ALLOWED_SECTION`, and `SPAWN_DENIED_SECTION`.

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
- `manual` — Agent explicitly calls `share(key, value)` when it decides something is worth sharing.
- `auto_conclusions` — Agent's final output is auto-copied to `shared:output:{agent_name}` namespace. Intermediate thoughts stay private.
- `full_transparency` — Same as `auto_conclusions` — final output is auto-published. Used for PM/tracker agents.

### How Agents Communicate

```python
class BlackboardClient:
    """Per-agent client for Redis-backed shared memory, pub/sub, and streams."""

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        redis: aioredis.Redis | None = None,
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.redis = redis or aioredis.from_url(REDIS_URL, decode_responses=True)
        self._stream_key = f"stream:{session_id}"
        self._notify_channel = f"notify:{session_id}"

    # ── Key-Value (Blackboard) ──────────────────────────────────

    async def write(self, key: str, value: Any) -> None:
        """Write to agent's private namespace."""
        full_key = f"blackboard:{self.session_id}:agent:{self.agent_id}:{key}"
        await self.redis.set(full_key, json.dumps(value))

    async def share(self, key: str, value: Any) -> None:
        """Publish to shared namespace + notify listeners via pub/sub."""
        full_key = f"blackboard:{self.session_id}:shared:{key}"
        await self.redis.set(full_key, json.dumps(value))
        await self.redis.publish(
            self._notify_channel,
            json.dumps({"agent": self.agent_id, "key": key, "event": "shared"}),
        )

    async def read_shared(self, pattern: str = "*") -> dict[str, Any]:
        """Read all shared findings matching a pattern."""
        full_pattern = f"blackboard:{self.session_id}:shared:{pattern}"
        keys = []
        async for key in self.redis.scan_iter(match=full_pattern):
            keys.append(key)
        result = {}
        for k in keys:
            raw = await self.redis.get(k)
            if raw:
                result[k] = json.loads(raw)
        return result

    async def read_mine(self, pattern: str = "*") -> dict[str, Any]:
        """Read own private memory."""
        full_pattern = f"blackboard:{self.session_id}:agent:{self.agent_id}:{pattern}"
        keys = []
        async for key in self.redis.scan_iter(match=full_pattern):
            keys.append(key)
        result = {}
        for k in keys:
            raw = await self.redis.get(k)
            if raw:
                result[k] = json.loads(raw)
        return result

    # ── Pub/Sub (Push Notifications) ────────────────────────────

    async def wait_for_key(self, key: str, timeout: int = BLACKBOARD_READ_TIMEOUT) -> Any | None:
        """Block until a specific shared key appears. Push-based, no polling."""
        full_key = f"blackboard:{self.session_id}:shared:{key}"
        existing = await self.redis.get(full_key)
        if existing:
            return json.loads(existing)

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._notify_channel)
        deadline = time.monotonic() + timeout
        try:
            async for message in pubsub.listen():
                if time.monotonic() > deadline:
                    return None
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    if data.get("key") == key:
                        raw = await self.redis.get(full_key)
                        return json.loads(raw) if raw else None
        finally:
            await pubsub.unsubscribe(self._notify_channel)
            await pubsub.aclose()

    # ── Streams (Task Dispatch) ─────────────────────────────────

    async def dispatch_task(self, target_agent_id: str, task: dict) -> None:
        """Send a task to a specific agent via Redis Stream."""
        await self.redis.xadd(
            self._stream_key,
            {
                "from": self.agent_id,
                "to": target_agent_id,
                "type": "task",
                "payload": json.dumps(task),
            },
        )

    async def consume_tasks(self) -> list[dict]:
        """Read tasks dispatched to this agent from the stream."""
        group = f"group:{self.agent_id}"
        try:
            await self.redis.xgroup_create(self._stream_key, group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists

        messages = await self.redis.xreadgroup(
            group, self.agent_id, {self._stream_key: ">"}, count=10
        )
        results = []
        if messages:
            for _stream, entries in messages:
                for _msg_id, fields in entries:
                    if fields.get("to") == self.agent_id:
                        results.append(json.loads(fields["payload"]))
        return results

    # ── Cleanup ─────────────────────────────────────────────────

    async def cleanup_session(self, session_id: str) -> None:
        """Remove all blackboard keys for a completed session."""
        pattern = f"blackboard:{session_id}:*"
        keys = []
        async for key in self.redis.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await self.redis.delete(*keys)
```

Artifact storage is handled separately by `memory/artifacts.py`, which provides `LocalObjectStore` (dev) and `S3ObjectStore` (production) backends via an `ObjectStore` protocol.

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
    "model": "azure/gpt-4o",
    "task": "Find best models for floor-plan-to-image generation",
    "success_criteria": "Ranked list of 3+ models with pros/cons",
    "output_format": "structured_report"
  },

  // Compiled prompt stored separately from blueprint
  "system_prompt_compiled": "...",

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
    "llm_calls": 3
  },

  // Token Usage (subtree) — this agent + ALL descendants, rolled up
  "subtree_token_usage": {
    "input_tokens": 18400,
    "output_tokens": 9200,
    "total_tokens": 27600,
    "llm_calls": 14,
    "agent_count": 3          // self + 2 children
  },

  // Results
  "final_output": null,       // Populated on completion
  "exit_reason": null,        // "completed" | "ttl_expired" | "budget_exhausted" | "killed_by_parent"
  "saved_state": null         // Populated on hibernation for resume
}
```

### Context Aggregation (Solves: Context Window Blow-up)

When child agents complete, their output is returned to the parent via `_handle_spawn()` in the tool node. Outputs are truncated inline — no separate aggregator class:

```python
async def _handle_spawn(args, parent_bp, spawner, deps) -> dict:
    child_bp = await spawner.maybe_spawn(parent_bp, args)
    if not child_bp:
        return {"tool": "spawn_agent", "status": "denied", "reason": "Reuse existing or budget exhausted."}

    child_output = await run_agent_graph(
        blueprint=child_bp,
        session_id=deps["session_id"],
        cost_tracker=deps["cost_tracker"],
        event_logger=deps["event_logger"],
        blackboard_redis=deps["blackboard"].redis,
        spawner=deps["spawner"],
        checkpointer=deps.get("checkpointer"),
    )

    # Truncate to 3000 chars so parent's context doesn't blow up
    summary = child_output[:3000] if len(child_output) <= 3000 else child_output[:3000] + "\n...[truncated]"
    return {"tool": "spawn_agent", "status": "completed", "agent_name": child_bp.name, "output": summary}
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
    """Manages TTL heartbeats, hibernation, and resurrection for a single agent."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        depth: int,
        ttl_seconds: int,
        max_steps: int,
        session_id: str,
        registry: AgentRegistry,
        event_logger: EventLogger,
        redis: aioredis.Redis,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.depth = depth
        self.ttl_seconds = ttl_seconds
        self.max_steps = max_steps
        self.session_id = session_id
        self.registry = registry
        self.event_logger = event_logger
        self.redis = redis

    async def start(self) -> None:
        """Initialize the heartbeat key in Redis."""
        await self.redis.setex(f"heartbeat:{self.agent_id}", self.ttl_seconds, "alive")

    async def check_alive(self) -> bool:
        """Check if the agent's heartbeat is still valid."""
        return bool(await self.redis.exists(f"heartbeat:{self.agent_id}"))

    async def step(self) -> int:
        """Called after each agent action. Checks TTL and step limit.
        Returns updated steps_taken."""
        if not await self.check_alive():
            await self.hibernate(reason="ttl_expired")
            raise AgentExpired(f"Agent {self.agent_id} TTL expired")

        steps = await self.registry.increment_steps(self.agent_id)
        if steps >= self.max_steps:
            await self.die(reason="max_steps_exceeded")
            raise AgentExpired(f"Agent {self.agent_id} exceeded max steps ({self.max_steps})")
        return steps

    async def extend_ttl(self, extra_seconds: int) -> None:
        """Extend the agent's TTL if it's making progress."""
        remaining = await self.redis.ttl(f"heartbeat:{self.agent_id}")
        if remaining > 0:
            await self.redis.expire(f"heartbeat:{self.agent_id}", remaining + extra_seconds)
            await self.event_logger.log_event(
                session_id=self.session_id, agent_id=self.agent_id,
                agent_name=self.agent_name, depth=self.depth,
                event_type=AGENT_TTL_EXTENDED,
                payload={"extra_seconds": extra_seconds, "new_remaining": remaining + extra_seconds},
            )

    async def hibernate(self, reason: str = "ttl_expired", saved_state: dict | None = None) -> None:
        """Put the agent to sleep — preserves state for resurrection."""
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(self.agent_id, status="hibernated", exit_reason=reason,
                                   saved_state=saved_state or {})
        await self.event_logger.log_event(
            session_id=self.session_id, agent_id=self.agent_id,
            agent_name=self.agent_name, depth=self.depth,
            event_type=AGENT_HIBERNATED, payload={"reason": reason},
        )

    async def complete(self, final_output: str) -> None:
        """Mark the agent as successfully completed."""
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(self.agent_id, status="completed",
                                   exit_reason="completed", final_output=final_output)

    async def die(self, reason: str = "killed") -> None:
        """Terminate the agent."""
        await self.redis.delete(f"heartbeat:{self.agent_id}")
        await self.registry.update(self.agent_id, status="dead", exit_reason=reason)
        await self.event_logger.log_event(
            session_id=self.session_id, agent_id=self.agent_id,
            agent_name=self.agent_name, depth=self.depth,
            event_type=AGENT_DIED, payload={"reason": reason},
        )
```

### Resurrection Protocol

```python
async def resurrect(
    agent_id: str,
    registry: AgentRegistry,
    event_logger: EventLogger,
    redis: aioredis.Redis,
    session_id: str,
    new_task: str | None = None,
    new_ttl: int | None = None,
) -> tuple[AgentBlueprint, str, dict]:
    """Resurrect a hibernated agent. Returns (blueprint, system_prompt, saved_state)."""
    record = await registry.find_by_id(agent_id)
    if not record or record["status"] != "hibernated":
        raise ValueError(f"Agent {agent_id} is not hibernated")

    bp_data = record["blueprint"]
    saved_state = record.get("saved_state", {})

    blueprint = AgentBlueprint(**{
        k: v for k, v in bp_data.items()
        if k in AgentBlueprint.__dataclass_fields__
    })

    if new_task:
        blueprint.task = new_task

    ttl = new_ttl or blueprint.ttl_seconds
    system_prompt = compile_system_prompt(blueprint)

    # Add resurrection context from RESURRECTION_CONTEXT_TEMPLATE
    resume_instruction = f"New additional task: {new_task}" if new_task else "Resume where you left off."
    resurrection_context = RESURRECTION_CONTEXT_TEMPLATE.format(
        partial_results=saved_state.get('partial_results', 'None'),
        last_step=saved_state.get('last_step', 'None'),
        resume_instruction=resume_instruction,
    )

    # Update registry and set heartbeat
    await registry.update(agent_id, status="alive", ttl_seconds=ttl,
                          expires_at=..., exit_reason=None)
    await redis.setex(f"heartbeat:{agent_id}", ttl, "alive")

    # Log resurrection event
    await event_logger.log_event(
        session_id=session_id, agent_id=agent_id,
        agent_name=blueprint.name, depth=blueprint.depth,
        event_type=AGENT_RESURRECTED,
        payload={"new_task": new_task, "previous_state_keys": list(saved_state.keys())},
    )

    return blueprint, system_prompt + resurrection_context, saved_state
```

---

## 5. Logging — The Immutable Event Journal

Every agent action produces an **event** appended to an immutable log. Nothing is ever deleted from this log.

### Event Schema

```python
@dataclass
class AgentEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())  # ISO 8601 string
    session_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    depth: int = 0
    event_type: str = ""        # See event types below
    payload: dict = field(default_factory=dict)
    parent_event_id: str | None = None  # For causal chain tracing
```

### Event Types

| Event Type | When | Payload |
|---|---|---|
| `agent.spawned` | Agent created | `{blueprint, parent_id, spawn_reason}` |
| `agent.thought` | LLM internal reasoning | `{thought_text, step_number}` |
| `agent.tool_call` | Tool invoked | `{tool_name, args, result_preview}` |
| `agent.tool_error` | Tool failed | `{tool_name, error}` |
| `agent.spawn_decision` | Decided to spawn (or not) | `{assessment: SpawnAssessment, decision, reason}` |
| `agent.blackboard_write` | Wrote to blackboard | `{namespace, key, value_preview}` |
| `agent.blackboard_read` | Read from blackboard | `{namespace, pattern, keys_returned}` |
| `agent.output` | Produced final output | `{output, status}` |
| `agent.hibernated` | TTL expired, going to sleep | `{reason}` |
| `agent.resurrected` | Woken from hibernation | `{new_task, previous_state_keys}` |
| `agent.died` | Terminated | `{reason}` |
| `agent.cost` | LLM call made | `{model, input_tokens, output_tokens, session_total_tokens}` |
| `agent.ttl_extended` | TTL extended | `{extra_seconds, new_remaining}` |

### Storage & Querying

```python
class EventLogger:
    """Async MongoDB-backed immutable event journal."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["events"]

    async def ensure_indexes(self):
        await self.collection.create_index("session_id")
        await self.collection.create_index("agent_id")
        await self.collection.create_index("event_type")
        await self.collection.create_index("timestamp")
        await self.collection.create_index(
            [("session_id", 1), ("agent_id", 1), ("timestamp", 1)]
        )

    async def log(self, event: AgentEvent) -> None:
        """Append an event to the journal. Never update or delete."""
        await self.collection.insert_one(asdict(event))

    async def log_event(
        self, session_id, agent_id, agent_name, depth, event_type, payload,
        parent_event_id=None,
    ) -> AgentEvent:
        """Convenience: build and log an event in one call."""
        event = AgentEvent(
            session_id=session_id, agent_id=agent_id, agent_name=agent_name,
            depth=depth, event_type=event_type, payload=payload,
            parent_event_id=parent_event_id,
        )
        await self.log(event)
        return event

    async def get_agent_trace(self, agent_id: str) -> list[dict]:
        """Full chronological trace of one agent's life."""
        cursor = self.collection.find({"agent_id": agent_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_session_events(self, session_id: str) -> list[dict]:
        """Everything that happened in a session — for post-mortem."""
        cursor = self.collection.find({"session_id": session_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_spawn_tree(self, session_id: str) -> list[dict]:
        """Get all spawn events for hierarchy reconstruction."""
        cursor = self.collection.find(
            {"session_id": session_id, "event_type": "agent.spawned"}
        ).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    async def get_cost_events(self, session_id: str) -> list[dict]:
        """Get all cost events for budget analysis."""
        cursor = self.collection.find(
            {"session_id": session_id, "event_type": "agent.cost"}
        ).sort("timestamp", 1)
        return await cursor.to_list(length=None)
```

### Post-Evaluation Dashboard (What You Can Answer)

From this log, you can answer:
- **How many agents were spawned?** Count `agent.spawned` events.
- **Was spawning efficient?** Compare agents that produced useful output vs. those that died without output.
- **What was the total token usage?** Sum all `agent.cost` events.
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

### Context Window Blow-up → Inline Output Truncation

```
Child agent outputs are returned to the parent as tool results.
Each child output → truncated to 3000 chars in _handle_spawn().
Full data stays on blackboard — parent can drill down via read_shared.
Additionally, agents with "auto_conclusions" or "full_transparency" share
  policy auto-publish their final output to the shared namespace.
```

### Tool Safety → Allowlist per Agent

```python
# Tools are allowlisted, not blocklisted.
# Each agent only gets the tools its blueprint specifies.
# Dangerous tools (file_delete, shell_exec) require:
#   1. Blueprint explicitly lists them
#   2. Agent depth <= RESTRICTED_TOOLS_MAX_DEPTH (default: 1)
#   3. Access validated before every tool call

RESTRICTED_TOOLS = {"shell_exec", "file_delete", "network_request_external"}
RESTRICTED_TOOLS_MAX_DEPTH = 1

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
```

### Cost Explosion → Token Budget + Model Tiering + Per-Agent Token Accounting

Two classes in `core/cost_tracker.py` handle all token accounting:

**`TokenUsage`** — A simple dataclass ledger that tracks `input_tokens`, `output_tokens`, `total_tokens`, and `llm_calls` for a single agent. Updated via `record(inp, out)` after each LLM call.

**`CostTracker`** — Session-wide singleton shared by all agents in a run. Three key methods:

| Method | When Called | What It Does |
|---|---|---|
| `charge()` | After every LLM call | 1. Adds tokens to in-memory `TokenUsage` for that agent. 2. Persists `token_usage` to MongoDB. 3. Logs an `agent.cost` event. 4. Raises `TokenCapExceeded` if session total hits `MAX_SESSION_TOKENS` (default: 500k). |
| `rollup_subtree()` | When an agent completes | Sums the agent's own tokens + all children's `subtree_token_usage` from MongoDB, writes the rolled-up total back. Parents can see "my entire subtree used X tokens." |
| `get_session_summary()` | At session end | Returns full breakdown: per-agent totals, session totals, `tokens_remaining`. |

Model tiering (cheap models for cheap tasks) keeps costs bounded — see Section 2. LLM concurrency is capped at `MAX_CONCURRENT_LLM_CALLS` (default: 3) via an asyncio semaphore in `core/agent.py`.

### Deadlocks → No Circular Dependencies by Design

```
Rule: Information flows DOWN (parent → child) and UP (child → parent).
Agents at the same depth NEVER wait on each other directly.
They communicate via blackboard + pub/sub (push-based, not polling).
If Agent A needs Agent B's output:
  → A calls wait_for_key("research:top_models", timeout=60)
  → Redis pub/sub pushes notification the instant B shares the key
  → No CPU-wasting polls

wait_timeout = BLACKBOARD_READ_TIMEOUT (default: 60 seconds)
on_timeout → report partial results to parent, let parent decide
```

### Spawn Race Conditions → Redis Distributed Locks

```python
class SpawnLock:
    """Async Redis-based distributed lock for safe spawn budget enforcement."""

    def __init__(self, redis: aioredis.Redis | None = None):
        self.redis = redis or aioredis.from_url(REDIS_URL, decode_responses=True)

    @asynccontextmanager
    async def acquire(self, session_id: str, timeout: float = 5.0) -> AsyncGenerator[bool, None]:
        """Context manager that acquires a distributed lock for spawning."""
        lock = self.redis.lock(
            f"spawn_lock:{session_id}",
            timeout=timeout,
            blocking_timeout=3.0,
        )
        acquired = await lock.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                await lock.release()
```

Used inside `Spawner._create_agent()`:

```python
async def _create_agent(self, parent, spec) -> AgentBlueprint | None:
    async with self.spawn_lock.acquire(self.session_id) as acquired:
        if not acquired:
            return None
        # Check global cap, parent budget, create child — all under lock
        ...
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
       ├── ResearchAgent graph (model=azure/gpt-4o, tools=[web_search], ttl=300s)
       ├── ArchitectAgent graph (model=azure/gpt-4o, tools=[code_execute], ttl=300s)
       └── PMAgent graph (model=azure/gpt-4o-mini, tools=[], share_policy=full_transparency)

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
   ├── Total tokens: ~27,000 (within 500k session cap)
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

---

## 9. Project Structure

```
Evolve/
├── core/
│   ├── genesis.py          # Genesis Agent — top-level orchestrator (LangGraph)
│   ├── agent.py            # LangGraph StateGraph: reason → tools → reason loop
│   │                        #   AgentState, reason_node, tool_node, should_continue
│   │                        #   build_agent_graph(), run_agent_graph()
│   ├── blueprint.py        # AgentBlueprint, MemoryScope, SpawnBudget + prompt compiler
│   ├── spawner.py          # Spawn logic: assessment, budget, reuse-first
│   ├── lifecycle.py        # TTL heartbeat, hibernate, resurrect, death
│   └── cost_tracker.py     # Token-based budget tracking, subtree rollup
│
├── memory/
│   ├── blackboard.py       # Redis KV + Pub/Sub + Streams + cleanup
│   ├── locks.py            # Redis distributed lock — spawn budget concurrency control
│   ├── registry.py         # MongoDB agent registry (DNA storage)
│   ├── artifacts.py        # Object storage client (Local/S3)
│   └── scoping.py          # MemoryScope helpers, namespace logic
│
├── tracing/
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
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_agent_graph.py
│   ├── test_blueprint.py
│   ├── test_lifecycle.py
│   ├── test_spawner.py
│   └── test_tool_registry.py
│
├── site/
│   ├── index.html          # Landing page
│   └── style.css           # Site styles
│
├── config.py               # Global caps, defaults, model routing, LiteLLM kwargs
├── main.py                 # CLI entry point
├── pyproject.toml          # Project metadata and build config
├── requirements.txt
├── README.md
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
