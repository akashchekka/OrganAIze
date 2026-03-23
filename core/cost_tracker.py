"""Token tracker — pure in-memory, per-agent token accounting."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from config import MAX_SESSION_TOKENS

logger = logging.getLogger("organaize.token_tracker")


@dataclass
class TokenUsage:
    """Token consumption ledger for a single agent."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def record(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += inp + out
        self.llm_calls += 1


class CostTracker:
    """Session-wide token tracker. Pure in-memory — no DB dependency."""

    def __init__(self, max_tokens: int = MAX_SESSION_TOKENS):
        self.max_tokens = max_tokens
        self.total_tokens_used: int = 0
        self.agent_usage: dict[str, TokenUsage] = {}

    def charge(
        self,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record a single LLM call's token usage."""
        self.total_tokens_used += input_tokens + output_tokens

        if agent_id not in self.agent_usage:
            self.agent_usage[agent_id] = TokenUsage()
        self.agent_usage[agent_id].record(input_tokens, output_tokens)

        if self.max_tokens > 0 and self.total_tokens_used >= self.max_tokens:
            raise TokenCapExceeded(
                f"Session token cap exceeded: {self.total_tokens_used:,} >= {self.max_tokens:,}"
            )

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.total_tokens_used)

    def get_session_summary(self) -> dict:
        """Return full token breakdown for the session."""
        return {
            "total_input_tokens": sum(u.input_tokens for u in self.agent_usage.values()),
            "total_output_tokens": sum(u.output_tokens for u in self.agent_usage.values()),
            "total_tokens": sum(u.total_tokens for u in self.agent_usage.values()),
            "total_llm_calls": sum(u.llm_calls for u in self.agent_usage.values()),
            "max_session_tokens": self.max_tokens,
            "tokens_remaining": self.tokens_remaining,
            "per_agent": {
                aid: asdict(usage) for aid, usage in self.agent_usage.items()
            },
        }


class TokenCapExceeded(Exception):
    """Raised when the session's token cap is exceeded."""
