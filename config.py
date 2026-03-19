"""Global configuration for the Evolve agentic OS."""

import os
from dataclasses import dataclass, field
from typing import Any


# ── Agent Limits ────────────────────────────────────────────────
MAX_GLOBAL_AGENTS = int(os.getenv("EVOLVE_MAX_AGENTS", "20"))
MAX_DEPTH = int(os.getenv("EVOLVE_MAX_DEPTH", "4"))
GENESIS_SPAWN_BUDGET = int(os.getenv("EVOLVE_GENESIS_BUDGET", "8"))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("EVOLVE_MAX_CONCURRENT_LLM", "3"))

# ── Session Budget ──────────────────────────────────────────────
MAX_SESSION_TOKENS = int(os.getenv("EVOLVE_MAX_SESSION_TOKENS", "500000"))  # Total token cap per session

# ── TTL Defaults ────────────────────────────────────────────────
DEFAULT_TTL_SECONDS = int(os.getenv("EVOLVE_DEFAULT_TTL", "300"))
DEFAULT_MAX_STEPS = int(os.getenv("EVOLVE_DEFAULT_MAX_STEPS", "30"))
BLACKBOARD_READ_TIMEOUT = int(os.getenv("EVOLVE_BB_READ_TIMEOUT", "60"))

# ── Spawn Assessment Thresholds ─────────────────────────────────
SPAWN_BREADTH_THRESHOLD = 0.6
SPAWN_PARALLELISM_THRESHOLD = 0.4
SPAWN_AUTO_THRESHOLD = 0.85

# ── MongoDB ─────────────────────────────────────────────────────
MONGO_URI = os.getenv("EVOLVE_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("EVOLVE_MONGO_DB", "evolve")

# ── Redis ───────────────────────────────────────────────────────
REDIS_URL = os.getenv("EVOLVE_REDIS_URL", "redis://localhost:6379/0")

# ── Object Storage ──────────────────────────────────────────────
OBJECT_STORE_TYPE = os.getenv("EVOLVE_OBJECT_STORE", "local")  # "s3" | "azure" | "local"
OBJECT_STORE_BUCKET = os.getenv("EVOLVE_OBJECT_BUCKET", "evolve-artifacts")
OBJECT_STORE_LOCAL_PATH = os.getenv("EVOLVE_OBJECT_LOCAL_PATH", "./artifacts")

# ── LLM Provider ────────────────────────────────────────────────
# LiteLLM routes based on model prefix:
#   OpenAI:       "gpt-4o"                 (uses OPENAI_API_KEY)
#   Azure OpenAI: "azure/gpt-4o"           (uses AZURE_API_KEY + AZURE_API_BASE)
#   Anthropic:    "claude-sonnet-4-20250514"       (uses ANTHROPIC_API_KEY)
#   Ollama:       "ollama/llama3"           (uses OLLAMA_API_BASE)
#   Any other:    See https://docs.litellm.ai/docs/providers

DEFAULT_LLM_MODEL = os.getenv("EVOLVE_DEFAULT_MODEL", "azure/gpt-4o")

# Provider-specific settings (LiteLLM reads these env vars automatically)
# Set these in .env — no code changes needed to switch providers.
# OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.

# Optional: explicit overrides (only needed if env vars aren't set)
LLM_API_KEY = os.getenv("EVOLVE_LLM_API_KEY", "")       # Fallback API key
LLM_API_BASE = os.getenv("EVOLVE_LLM_API_BASE", "")     # Custom endpoint (Ollama, vLLM, etc.)
LLM_API_VERSION = os.getenv("EVOLVE_LLM_API_VERSION", "")  # Azure API version

# ── Role → Default Model Mapping ───────────────────────────────
ROLE_MODEL_DEFAULTS: dict[str, str] = {
    "orchestrator": "azure/gpt-4o",
    "researcher":   "azure/gpt-4o",
    "engineer":     "azure/gpt-4o",
    "critic":       "azure/gpt-4o-mini",
    "qa":           "azure/gpt-4o-mini",
    "summarizer":   "azure/gpt-4o-mini",
    "pm":           "azure/gpt-4o-mini",
    "synthesizer":  "azure/gpt-4o",
}

# ── Restricted Tools ────────────────────────────────────────────
RESTRICTED_TOOLS = {"shell_exec", "file_delete", "network_request_external"}
RESTRICTED_TOOLS_MAX_DEPTH = 1  # Only agents at depth <= this can use restricted tools

def get_llm_kwargs(model: str | None = None) -> dict[str, Any]:
    """Build provider-aware kwargs for litellm.acompletion().

    LiteLLM auto-detects the provider from the model prefix:
      - "gpt-4o"          → OpenAI (reads OPENAI_API_KEY)
      - "azure/gpt-4o"    → Azure  (reads AZURE_API_KEY + AZURE_API_BASE)
      - "claude-sonnet"   → Anthropic (reads ANTHROPIC_API_KEY)
      - "ollama/llama3"   → Ollama (reads OLLAMA_API_BASE or EVOLVE_LLM_API_BASE)

    Only adds api_key/api_base if explicitly set via EVOLVE_LLM_* overrides.
    Otherwise, LiteLLM reads the standard env vars for each provider.
    """
    kwargs: dict[str, Any] = {"model": model or DEFAULT_LLM_MODEL}

    if LLM_API_KEY:
        kwargs["api_key"] = LLM_API_KEY
    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE
    if LLM_API_VERSION:
        kwargs["api_version"] = LLM_API_VERSION

    return kwargs
