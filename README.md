# Evolve — Self-Evolving Agentic AI

A system where AI agents **evolve** to solve problems — spawning specialists, sharing knowledge, and learning from experience, just like humans do.

## What It Does

Give Evolve a goal. A Genesis Agent decomposes it, spawns specialist sub-agents, and synthesizes their outputs. Each agent:

- **Self-assesses** whether it needs help (cognitive load detection)
- **Spawns specialists** with tailored personas, tools, and expertise
- **Communicates** via a shared blackboard (Redis pub/sub — no polling)
- **Lives and dies** with TTL heartbeats, hibernation, and resurrection
- **Tracks every action** in an immutable event journal

```
User Goal
   └── Genesis Agent (Orchestrator)
          ├── Spawns → ResearchAgent (finds best models)
          ├── Spawns → ArchitectAgent (designs the system)
          │     ├── Spawns → FrontendAgent
          │     └── Spawns → BackendAgent
          └── Synthesizes all outputs into final response
```

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB running on `localhost:27017`
- Redis running on `localhost:6379`
- An OpenAI API key (or any LiteLLM-compatible provider)

### Setup

```bash
git clone <repo-url> && cd Evolve
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key
```

### Run

```bash
python main.py "Build an app that converts floor plans into realistic images"
```

With options:

```bash
python main.py --budget 2.00 --model gpt-4o "Your goal here"
```

### Output

```
🧬 Evolve — Starting session
   Goal:   Build an app that converts floor plans into realistic images
   Budget: $5.00
   Model:  gpt-4o

======================================================================
EVOLVE SESSION COMPLETE
======================================================================

Session ID: a1b2c3d4-...
Agents spawned: 6

--- Token Summary ---
Total input tokens:  18,400
Total output tokens: 9,200
Total tokens:        27,600
Total LLM calls:     14
Total cost:          $0.0730
Budget remaining:    $4.9270

--- Agent Tree ---
[✓] Genesis-Orchestrator (role=orchestrator, tokens=6,000, cost=$0.0180)
  ├── [✓] ResearchAgent-CV (role=researcher, tokens=4,200, cost=$0.0120)
  ├── [✓] ArchitectAgent (role=engineer, tokens=8,000, cost=$0.0230)
  │   ├── [✓] FrontendAgent (role=engineer, tokens=3,200, cost=$0.0090)
  │   └── [✓] BackendAgent (role=engineer, tokens=6,000, cost=$0.0110)
  └── [✓] PMAgent (role=pm, tokens=500, cost=$0.0000)
```

## Architecture

See [DESIGN.md](DESIGN.md) for the full architecture document.

### Infrastructure

| Component | Tech | Role |
|---|---|---|
| **Agent Runtime** | LangGraph `StateGraph` | `reason → tools → reason` loop with checkpointing |
| **LLM** | LiteLLM | Unified API across OpenAI, Anthropic, local models |
| **Agent Registry** | MongoDB | Blueprints, lineage, token usage, event log |
| **Blackboard** | Redis | Shared memory (KV), pub/sub, streams, Redlock, heartbeats |
| **Artifacts** | S3 / Azure Blob / MinIO | Large outputs (images, code, datasets) |

### Key Concepts

- **AgentBlueprint** — The DNA of an agent: persona, expertise, tools, model, TTL, spawn budget
- **Spawn Budget** — Depth limit (4), budget halving per level, global cap (20)
- **Blackboard** — Namespaced Redis: `shared:*` (all agents), `agent:{id}:*` (private), `team:{tag}:*` (group)
- **Event Journal** — Immutable MongoDB log of every thought, tool call, spawn, and cost

## Project Structure

```
Evolve/
├── core/                   # Agent runtime (LangGraph)
│   ├── agent.py            # StateGraph: reason → tools → reason
│   ├── genesis.py          # Top-level orchestrator
│   ├── blueprint.py        # AgentBlueprint + prompt compiler
│   ├── spawner.py          # Spawn assessment, budget, reuse-first
│   ├── lifecycle.py        # TTL, hibernate, resurrect
│   └── cost_tracker.py     # Per-agent token tracking + rollup
├── memory/                 # Storage layer
│   ├── registry.py         # MongoDB agent registry
│   ├── blackboard.py       # Redis KV + pub/sub + streams
│   ├── locks.py            # Redlock for spawn concurrency
│   ├── artifacts.py        # Object storage (S3/local)
│   └── scoping.py          # Namespace logic
├── tracing/                # Observability
│   ├── event_logger.py     # Immutable event journal
│   ├── event_types.py      # Event type constants
│   └── trace.py            # Spawn tree + critical path
├── prompts/                # All prompt templates
│   ├── genesis.py          # Genesis persona
│   ├── spawner.py          # Assessment + decomposition
│   ├── agent.py            # System prompt template
│   └── lifecycle.py        # Resurrection context
├── tools/                  # Tool definitions + access control
│   └── tool_registry.py
├── tests/                  # Unit tests (88 tests)
│   ├── conftest.py         # Shared fixtures + mocks
│   ├── test_blueprint.py
│   ├── test_config.py
│   ├── test_cost_tracker.py
│   ├── test_lifecycle.py
│   ├── test_spawner.py
│   ├── test_tool_registry.py
│   ├── test_agent_graph.py
│   └── website_content.yaml
├── custom_loop/            # Legacy while-loop runtime (preserved)
├── config.py               # Global configuration
├── main.py                 # CLI entry point
├── requirements.txt
├── pyproject.toml          # Pytest configuration
├── .env.example
├── .gitignore
├── DESIGN.md               # Full architecture document
└── README.md               # This file
```

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM provider API key |
| `EVOLVE_MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `EVOLVE_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `EVOLVE_MAX_AGENTS` | `20` | Global agent cap per session |
| `EVOLVE_MAX_DEPTH` | `4` | Max agent tree depth |
| `EVOLVE_GENESIS_BUDGET` | `8` | Max children for genesis agent |
| `EVOLVE_SESSION_BUDGET_USD` | `5.00` | Session cost cap in USD |
| `EVOLVE_DEFAULT_MODEL` | `gpt-4o` | Default LLM model |
| `EVOLVE_DEFAULT_TTL` | `300` | Agent time-to-live (seconds) |
| `EVOLVE_MAX_CONCURRENT_LLM` | `3` | Max parallel LLM calls |

## License

MIT
