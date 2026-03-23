"""Prompt templates — re-exported for convenience."""

from prompts.genesis import GENESIS_PERSONA
from prompts.agent import (
    AGENT_SYSTEM_PROMPT_TEMPLATE,
    SPAWN_ALLOWED_SECTION,
    SPAWN_DENIED_SECTION,
)

__all__ = [
    "GENESIS_PERSONA",
    "AGENT_SYSTEM_PROMPT_TEMPLATE",
    "SPAWN_ALLOWED_SECTION",
    "SPAWN_DENIED_SECTION",
]
