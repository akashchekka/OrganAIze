# OrganAIze — Self-Organizing Agentic AI

A system where AI agents **self-organize** to solve problems — spawning specialists, coordinating in parallel, and synthesizing results.

## What It Does

Give OrganAIze a goal. A Genesis Agent decomposes it, spawns specialist sub-agents, and synthesizes their outputs. Each agent:

- **Spawns specialists** with tailored personas, tools, and expertise
- **Runs in parallel** via `asyncio.gather` when the LLM decides agents are independent
- **Returns results directly** — child outputs flow up the tree as tool messages
- **Tracks every token** with per-agent accounting and session-wide caps

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
- An API key for any LiteLLM-compatible provider (OpenAI, Azure, Anthropic, Ollama)

### Setup

```bash
git clone <repo-url> && cd OrganAIze
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
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
python main.py --max-tokens 100000 --model gpt-4o "Your goal here"
```

### Output

Session output is written to `output/<session_id>/`:
- `session.json` — metadata, token summary, agent count
- `output.md` — final synthesized markdown

```
OrganAIze - Starting session
   Goal:       Create a comprehensive report on quantum computing
   Max tokens: 100,000
   Model:      azure/gpt-4o

11:35:31 | genesis  | Spawning child QuantumHardwareResearcher (researcher, depth=1)
11:35:31 | genesis  | Spawning child QuantumAlgorithmsExplainer (summarizer, depth=1)
11:35:31 | genesis  | Spawning child QuantumCybersecurityAnalyst (critic, depth=1)
11:35:31 | genesis  | Spawning child QuantumAdvantagePredictor (pm, depth=1)
                      ^ All 4 spawned in parallel

======================================================================
ORGANAIZE SESSION COMPLETE
======================================================================

Session ID: 6cde816c-391e-46c7-8c6b-7bc5cd18edd9
Agents spawned: 5

--- Token Summary ---
Total input tokens:  23,839
Total output tokens: 6,624
Total tokens:        30,463
Total LLM calls:     11
Tokens remaining:    69,537

--- Agent Tree ---
[✓] Genesis-Orchestrator (role=orchestrator, tokens=26,350)
  ├── [✓] QuantumHardwareResearcher (role=researcher, tokens=1,094)
  ├── [✓] QuantumAlgorithmsExplainer (role=summarizer, tokens=922)
  ├── [✓] QuantumCybersecurityAnalyst (role=critic, tokens=1,179)
  └── [✓] QuantumAdvantagePredictor (role=pm, tokens=918)

Total time: 38 seconds (4 agents ran in parallel)
```

## Architecture

See [DESIGN.md](DESIGN.md) for the full architecture document.

### Infrastructure

| Component | Tech | Role |
|---|---|---|
| **Agent Runtime** | LangGraph `StateGraph` | `reason → tools → reason` loop with checkpointing |
| **LLM** | LiteLLM | Unified API across OpenAI, Anthropic, Azure, Ollama |
| **Token Tracking** | In-memory `CostTracker` | Per-agent token accounting, session cap |
| **Spawning** | In-memory `Spawner` | Budget enforcement, depth/global caps |
| **Output** | File system | `output/<session_id>/session.json` + `output.md` |

### Key Concepts

- **AgentBlueprint** — The DNA of an agent: persona, expertise, tools, model, spawn budget
- **Spawn Budget** — Depth limit (4), budget halving per level, global cap (20)
- **Direct Output Flow** — Child outputs return to parents as tool messages, no shared memory
- **Parallel Spawning** — LLM sets `parallel: true` on spawn calls, system fans out via `asyncio.gather`

## Project Structure

```
OrganAIze/
├── core/                   # Agent runtime (LangGraph)
│   ├── agent.py            # StateGraph: reason → tools → reason
│   ├── genesis.py          # Top-level orchestrator + file output
│   ├── blueprint.py        # AgentBlueprint + prompt compiler
│   ├── spawner.py          # Budget enforcement, agent creation
│   └── cost_tracker.py     # Per-agent token tracking
├── prompts/                # All prompt templates
│   ├── genesis.py          # Genesis persona
│   └── agent.py            # System prompt template
├── tools/                  # Tool definitions + access control
│   └── tool_registry.py
├── tests/                  # Unit tests (49 tests)
│   ├── conftest.py         # Shared fixtures
│   ├── test_agent_graph.py
│   ├── test_blueprint.py
│   ├── test_spawn_modes.py
│   ├── test_spawner.py
│   └── test_tool_registry.py
├── site/                   # Landing page
│   ├── index.html
│   └── style.css
├── config.py               # Global caps, defaults, model routing
├── main.py                 # CLI entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
├── DESIGN.md               # Full architecture document
└── README.md               # This file
```

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `AZURE_API_KEY` / `OPENAI_API_KEY` | — | LLM provider API key |
| `ORGANAIZE_DEFAULT_MODEL` | `azure/gpt-4o` | Default LLM model |
| `ORGANAIZE_MAX_AGENTS` | `20` | Global agent cap per session |
| `ORGANAIZE_MAX_DEPTH` | `4` | Max agent tree depth |
| `ORGANAIZE_GENESIS_BUDGET` | `8` | Max children for genesis agent |
| `ORGANAIZE_MAX_SESSION_TOKENS` | `500000` | Session token cap |
| `ORGANAIZE_MAX_CONCURRENT_LLM` | `3` | Max parallel LLM calls |
| `ORGANAIZE_DEFAULT_MAX_STEPS` | `30` | Max LLM reasoning steps per agent |
| `ORGANAIZE_OUTPUT_DIR` | `./output` | Session output directory |

## License

MIT
