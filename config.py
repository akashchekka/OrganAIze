"""Global configuration for OrganAIze."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ── Agent Limits ────────────────────────────────────────────────
MAX_GLOBAL_AGENTS = int(os.getenv("ORGANAIZE_MAX_AGENTS", "20"))
MAX_DEPTH = int(os.getenv("ORGANAIZE_MAX_DEPTH", "4"))
GENESIS_SPAWN_BUDGET = int(os.getenv("ORGANAIZE_GENESIS_BUDGET", "8"))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("ORGANAIZE_MAX_CONCURRENT_LLM", "3"))

# ── Session Budget ──────────────────────────────────────────────
MAX_SESSION_TOKENS = int(os.getenv("ORGANAIZE_MAX_SESSION_TOKENS", "500000"))  # Total token cap per session

# ── Agent Steps ─────────────────────────────────────────────────
DEFAULT_MAX_STEPS = int(os.getenv("ORGANAIZE_DEFAULT_MAX_STEPS", "30"))

# ── Output Directory ────────────────────────────────────────────
OUTPUT_DIR = os.getenv("ORGANAIZE_OUTPUT_DIR", "./output")

# ── LLM Provider ────────────────────────────────────────────────
# LiteLLM routes based on model prefix:
#   OpenAI:       "gpt-4o"                 (uses OPENAI_API_KEY)
#   Azure OpenAI: "azure/gpt-4o"           (uses AZURE_API_KEY + AZURE_API_BASE)
#   Anthropic:    "claude-sonnet-4-20250514"       (uses ANTHROPIC_API_KEY)
#   Ollama:       "ollama/llama3"           (uses OLLAMA_API_BASE)
#   Any other:    See https://docs.litellm.ai/docs/providers

DEFAULT_LLM_MODEL = os.getenv("ORGANAIZE_DEFAULT_MODEL", "azure/gpt-4o")

# Provider-specific settings (LiteLLM reads these env vars automatically)
# Set these in .env — no code changes needed to switch providers.
# OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.

# Optional: explicit overrides (only needed if env vars aren't set)
LLM_API_KEY = os.getenv("ORGANAIZE_LLM_API_KEY", "")       # Fallback API key
LLM_API_BASE = os.getenv("ORGANAIZE_LLM_API_BASE", "")     # Custom endpoint (Ollama, vLLM, etc.)
LLM_API_VERSION = os.getenv("ORGANAIZE_LLM_API_VERSION", "")  # Azure API version

# ── Role → Default Model Mapping ───────────────────────────────
# All roles use DEFAULT_LLM_MODEL unless you override specific roles here.
ROLE_MODEL_DEFAULTS: dict[str, str] = {
    "orchestrator": DEFAULT_LLM_MODEL,
    "researcher":   DEFAULT_LLM_MODEL,
    "engineer":     DEFAULT_LLM_MODEL,
    "critic":       DEFAULT_LLM_MODEL,
    "qa":           DEFAULT_LLM_MODEL,
    "summarizer":   DEFAULT_LLM_MODEL,
    "pm":           DEFAULT_LLM_MODEL,
    "synthesizer":  DEFAULT_LLM_MODEL,
}

# ── Restricted Tools ────────────────────────────────────────────
RESTRICTED_TOOLS = {"shell_exec", "file_delete", "network_request_external", "code_execute"}
RESTRICTED_TOOLS_MAX_DEPTH = 1  # Only agents at depth <= this can use restricted tools

def get_llm_kwargs(model: str | None = None) -> dict[str, Any]:
    """Build provider-aware kwargs for litellm.acompletion().

    LiteLLM auto-detects the provider from the model prefix:
      - "gpt-4o"          → OpenAI (reads OPENAI_API_KEY)
      - "azure/gpt-4o"    → Azure  (reads AZURE_API_KEY + AZURE_API_BASE)
      - "claude-sonnet"   → Anthropic (reads ANTHROPIC_API_KEY)
      - "ollama/llama3"   → Ollama (reads OLLAMA_API_BASE or ORGANAIZE_LLM_API_BASE)

    Only adds api_key/api_base if explicitly set via ORGANAIZE_LLM_* overrides.
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
