"""Prompt templates — re-exported for convenience."""

from prompts.genesis import GENESIS_PERSONA
from prompts.spawner import ASSESSMENT_PROMPT, DECOMPOSITION_PROMPT
from prompts.agent import (
    AGENT_SYSTEM_PROMPT_TEMPLATE,
    SPAWN_ALLOWED_SECTION,
    SPAWN_DENIED_SECTION,
)
from prompts.lifecycle import RESURRECTION_CONTEXT_TEMPLATE

__all__ = [
    "GENESIS_PERSONA",
    "ASSESSMENT_PROMPT",
    "DECOMPOSITION_PROMPT",
    "AGENT_SYSTEM_PROMPT_TEMPLATE",
    "SPAWN_ALLOWED_SECTION",
    "SPAWN_DENIED_SECTION",
    "RESURRECTION_CONTEXT_TEMPLATE",
]
