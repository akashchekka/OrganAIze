"""Cost tracker with per-agent token accounting and subtree rollup."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from config import SESSION_BUDGET_USD, calculate_cost
from tracing.event_logger import EventLogger
from tracing.event_types import AGENT_COST
from memory.registry import AgentRegistry

logger = logging.getLogger("evolve.cost_tracker")


@dataclass
class TokenUsage:
    """Token consumption ledger for a single agent."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def record(self, model: str, inp: int, out: int) -> float:
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += inp + out
        self.llm_calls += 1
        cost = calculate_cost(model, inp, out)
        self.cost_usd += cost
        return cost


class CostTracker:
    """Session-wide cost tracker with per-agent ledgers and subtree rollup."""

    def __init__(
        self,
        registry: AgentRegistry,
        event_logger: EventLogger,
        session_id: str,
        budget_usd: float = SESSION_BUDGET_USD,
    ):
        self.registry = registry
        self.event_logger = event_logger
        self.session_id = session_id
        self.budget = budget_usd
        self.spent: float = 0.0
        self.agent_usage: dict[str, TokenUsage] = {}

    async def charge(
        self,
        agent_id: str,
        agent_name: str,
        depth: int,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a single LLM call's cost. Persists to MongoDB and event log."""
        cost = calculate_cost(model, input_tokens, output_tokens)
        self.spent += cost

        # Per-agent tracking
        if agent_id not in self.agent_usage:
            self.agent_usage[agent_id] = TokenUsage()
        self.agent_usage[agent_id].record(model, input_tokens, output_tokens)

        # Persist to agent registry
        await self.registry.update(
            agent_id, token_usage=asdict(self.agent_usage[agent_id])
        )

        # Log cost event
        await self.event_logger.log_event(
            session_id=self.session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            depth=depth,
            event_type=AGENT_COST,
            payload={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "session_total_usd": self.spent,
            },
        )

        # Budget enforcement
        if self.spent >= self.budget:
            raise BudgetExhausted(
                f"Session budget exhausted: ${self.spent:.4f} >= ${self.budget:.2f}"
            )

        return cost

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    @property
    def budget_pct_used(self) -> float:
        return (self.spent / self.budget * 100) if self.budget > 0 else 100.0

    async def rollup_subtree(self, agent_id: str) -> dict:
        """Roll up token usage from children into this agent's subtree_token_usage."""
        agent = await self.registry.find_by_id(agent_id)
        if not agent:
            return {}

        own = self.agent_usage.get(agent_id, TokenUsage())
        children_ids = agent.get("children", [])

        children_subtrees = []
        for cid in children_ids:
            child = await self.registry.find_by_id(cid)
            if child:
                children_subtrees.append(
                    child.get("subtree_token_usage", {})
                )

        subtree = {
            "input_tokens": own.input_tokens + sum(c.get("input_tokens", 0) for c in children_subtrees),
            "output_tokens": own.output_tokens + sum(c.get("output_tokens", 0) for c in children_subtrees),
            "total_tokens": own.total_tokens + sum(c.get("total_tokens", 0) for c in children_subtrees),
            "llm_calls": own.llm_calls + sum(c.get("llm_calls", 0) for c in children_subtrees),
            "cost_usd": own.cost_usd + sum(c.get("cost_usd", 0) for c in children_subtrees),
            "agent_count": 1 + sum(c.get("agent_count", 1) for c in children_subtrees),
        }

        await self.registry.update(agent_id, subtree_token_usage=subtree)
        return subtree

    def get_session_summary(self) -> dict:
        """Return full token breakdown for the session."""
        return {
            "total_input_tokens": sum(u.input_tokens for u in self.agent_usage.values()),
            "total_output_tokens": sum(u.output_tokens for u in self.agent_usage.values()),
            "total_tokens": sum(u.total_tokens for u in self.agent_usage.values()),
            "total_llm_calls": sum(u.llm_calls for u in self.agent_usage.values()),
            "total_cost_usd": self.spent,
            "budget_usd": self.budget,
            "budget_remaining_usd": self.budget_remaining,
            "per_agent": {
                aid: asdict(usage) for aid, usage in self.agent_usage.items()
            },
        }


class BudgetExhausted(Exception):
    """Raised when the session's USD budget is fully consumed."""
