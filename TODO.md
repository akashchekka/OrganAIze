# Evolve — TODOs

## Completed

- [x] **Core architecture design** — spawning, traits, communication, lifecycle, logging
- [x] **Infrastructure layer** — MongoDB registry, Redis blackboard (KV + pub/sub + streams + locks), object storage
- [x] **AgentBlueprint + prompt compiler** — structured blueprint schema, system prompt generation
- [x] **Cost tracker** — per-agent token accounting, subtree rollup, session budget enforcement
- [x] **Event logger** — immutable event journal with causal chain tracing
- [x] **Spawner** — spawn assessment, reuse-first, budget halving, depth cap, global cap, Redlock
- [x] **Agent lifecycle** — TTL heartbeat, hibernate, resurrect, death state machine
- [x] **Tool registry** — allowlist-based access control, restricted tools by depth
- [x] **LangGraph refactor** — `StateGraph` with `reason → tools → reason` loop, checkpointing via `MemorySaver`
- [x] **Prompt separation** — domain-specific prompt files in `prompts/`
- [x] **Custom loop preserved** — legacy while-loop approach in `test/custom_loop/`

## P0 — Latency & Performance

- [x] **Parallel child execution** — LLM controls via `parallel` flag on `spawn_agent`. `parallel: true` (default) fans out via `asyncio.gather`; `parallel: false` runs sequentially. Non-spawn tools always run sequentially.
- [ ] **Background event logging** — push `event_logger.log_event` calls to an async queue instead of awaiting on the hot path
- [ ] **Batch MongoDB writes in `charge()`** — buffer `registry.update` + `event_logger.log_event` and flush periodically, not per LLM call
- [ ] **Redis pipeline for `read_shared`/`read_mine`** — replace `scan_iter` + N individual `GET`s with `SCAN` + `MGET` pipeline
- [ ] **Batch `rollup_subtree` reads** — replace N+1 sequential `find_by_id` calls with a single `$in` query for all children
- [ ] **Cache compiled agent graph** — `build_agent_graph()` returns the same topology every time; build once, reuse across spawns
- [ ] **Priority-aware LLM semaphore** — replace global `Semaphore(3)` with depth-aware queuing so parent agents aren't starved by children

## P1 — Production Hardening

- [ ] **LLM resilience** — retry with exponential backoff, model fallback chain (gpt-4o → claude-sonnet → gpt-4o-mini)
- [ ] **Error recovery** — circuit breakers per agent, agent-level isolation (one agent crash doesn't kill session)
- [ ] **Sandboxed code execution** — replace subprocess with container-based sandbox (gVisor/Firecracker) or sandboxed interpreter API
- [ ] **Rate limiting** — per-model token/minute tracking matching provider rate limits
- [ ] **Auth & multi-tenancy** — session isolation, API auth, per-tenant budget caps
- [ ] **Real tool implementations** — wire up SerpAPI/Tavily for web_search, PDF parser for read_paper, etc.
- [ ] **LangGraph PostgreSQL checkpointer** — replace MemorySaver with persistent checkpointer for crash recovery across restarts

## P2 — Dashboard & Observability

- [ ] **Observability** — OpenTelemetry traces, structured logging, Prometheus metrics for agent count, cost, latency
- [ ] **Streamlit dashboard** — real-time session monitoring
- [ ] **Spawn tree visualization** — interactive tree from event log spawn events
- [ ] **Cost report views** — per-agent, per-session, per-model cost breakdowns
- [ ] **Token usage heatmap** — which agents consumed the most tokens and why
- [ ] **Replay mode** — step through a completed session event-by-event for post-mortem
- [ ] **Scaling** — worker pool (Celery/Dramatiq) or containerized agents for horizontal scaling

## P3 — Dynamic Skill System

### Phase 1 — Skill DB + Curated Skills
- [ ] Design MongoDB `skills` collection schema (name, description, keywords, embedding, trust_level, body, assets, stats)
- [ ] Build `SkillRegistry` class — CRUD, keyword search, embedding similarity search
- [ ] Build `Skill` dataclass — parse SKILL.md frontmatter + body + asset catalog
- [ ] Build `SkillMatcher` — three-tier lookup: in-memory cache → MongoDB → fallback
- [ ] Wire skill matching into `Spawner._create_agent()` — enrich blueprint with matched skill
- [ ] Handle skill assets: templates (inject via file_write), scripts (execute via code_execute), examples (append to system prompt)
- [ ] Skill asset sandboxing — scripts run in subprocess with timeout, no network, workspace-restricted
- [ ] Skill trust levels — `verified` | `community` | `discovered`; depth > 1 agents only use `verified`
- [ ] Ship 3-5 curated starter skills (e.g., research, web-app-scaffold, data-analysis, code-review)
- [ ] Add SKILL.md prompts to `prompts/` for skill-enriched system prompts

### Phase 2 — Mid-Run Skill Discovery
- [ ] Add `search_skill` tool — agents can search the skill DB during execution (not just at spawn)
- [ ] Add `load_skill` tool — agent can load a skill mid-run, enriching its own system prompt
- [ ] Rate limiting on skill lookups — max 3 skill searches per agent per session
- [ ] Skill loading logged in event journal (`agent.skill_loaded` event type)

### Phase 3 — Web Search Skill Discovery
- [ ] Web search fallback when skill DB has no match — agent searches web for domain knowledge
- [ ] Skill ingestion pipeline — extract SKILL.md-compatible structure from web results
- [ ] Trust gating — web-discovered skills marked as `discovered`, require validation before use
- [ ] LLM-based skill quality check — evaluate discovered skill for accuracy, relevance, safety
- [ ] Deduplication — check if a semantically similar skill already exists before ingesting

### Phase 4 — Agent-Authored Skills
- [ ] After successful novel task, agent proposes a new skill based on its experience
- [ ] Skill generation prompt — agent produces SKILL.md with workflow, tools used, key patterns, code templates
- [ ] Auto-saved to DB with `trust_level="discovered"` and `created_by=agent:{id}`
- [ ] Skill rating system — parent agent rates child's skill contribution after use
- [ ] Human review workflow — discovered/agent-authored skills queued for promotion to `community` or `verified`
- [ ] Skill deprecation — flag skills with low success rates or old last-validated timestamps
- [ ] Skill versioning — track version history, allow rollback
- [ ] Skill usage analytics — which skills are loaded most, success rates, agent ratings
