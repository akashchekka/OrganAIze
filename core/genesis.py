"""Genesis Agent — the top-level orchestrator that receives user goals.

Uses LangGraph for the agent runtime. The Genesis agent is itself a graph node
that can spawn child agent graphs via the spawn_agent tool.

All output is written to a session folder under ``output/<session_id>/``:
  - session.json  — blueprint, token summary, agent tree
  - output.md     — final synthesized markdown output
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_STEPS,
    GENESIS_SPAWN_BUDGET,
    MAX_DEPTH,
    MAX_GLOBAL_AGENTS,
    MAX_SESSION_TOKENS,
    OUTPUT_DIR,
)
from core.agent import run_agent_graph
from core.blueprint import AgentBlueprint, SpawnBudget
from core.cost_tracker import CostTracker
from core.spawner import Spawner
from prompts.genesis import GENESIS_PERSONA

logger = logging.getLogger("organaize.genesis")


class GenesisAgent:
    """Top-level orchestrator. Receives a user goal, decomposes, spawns, synthesizes.

    Pure in-memory — no MongoDB, no Redis. Session output is written to disk.
    """

    def __init__(
        self,
        max_session_tokens: int = MAX_SESSION_TOKENS,
        model: str = DEFAULT_LLM_MODEL,
    ):
        self.session_id = str(uuid.uuid4())
        self.model = model
        self.max_session_tokens = max_session_tokens

        self.cost_tracker = CostTracker(max_tokens=max_session_tokens)
        self.spawner = Spawner(session_id=self.session_id)
        self._checkpointer = MemorySaver()

    async def run(self, user_goal: str) -> dict:
        """Execute the full OrganAIze pipeline for a user goal.

        Returns a dict with:
        - output: The final synthesized response
        - session_id: The session identifier
        - token_summary: Per-agent and total token usage
        - agents_spawned: Number of agents that were created
        """
        logger.info("Session %s started | goal='%s' | model=%s | max_tokens=%d",
                     self.session_id, user_goal[:100], self.model, self.max_session_tokens)

        # 1. Create the Genesis agent blueprint
        genesis_blueprint = AgentBlueprint(
            name="Genesis-Orchestrator",
            role="orchestrator",
            persona=GENESIS_PERSONA,
            expertise=["decomposition", "orchestration", "synthesis"],
            tools_allowed=["spawn_agent", "web_search"],
            tools_denied=[],
            model=self.model,
            max_steps=DEFAULT_MAX_STEPS * 2,
            spawn_budget=SpawnBudget(
                max_children=GENESIS_SPAWN_BUDGET,
                max_depth_remaining=MAX_DEPTH,
                remaining_global=MAX_GLOBAL_AGENTS,
            ),
            task=user_goal,
            success_criteria="Fully address the user's goal by orchestrating specialist agents and synthesizing their outputs.",
            output_format="markdown",
            depth=0,
        )

        # 2. Run the genesis agent as a LangGraph
        logger.info("Running Genesis agent graph (id=%s)", genesis_blueprint.agent_id)
        output = await run_agent_graph(
            blueprint=genesis_blueprint,
            session_id=self.session_id,
            cost_tracker=self.cost_tracker,
            spawner=self.spawner,
            checkpointer=self._checkpointer,
        )

        # 3. Gather session summary
        token_summary = self.cost_tracker.get_session_summary()
        agents_spawned = self.spawner.agents_created + 1  # +1 for genesis itself

        logger.info("Session %s complete | agents=%d | tokens=%d",
                     self.session_id, agents_spawned, token_summary["total_tokens"])

        result = {
            "output": output,
            "session_id": self.session_id,
            "token_summary": token_summary,
            "agents_spawned": agents_spawned,
        }

        # 4. Write session output to disk
        self._write_session_output(result, user_goal)

        return result

    def _write_session_output(self, result: dict, user_goal: str) -> None:
        """Write session.json and output.md to ``output/<session_id>/``."""
        session_dir = Path(OUTPUT_DIR) / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # session.json — metadata + token summary
        session_meta = {
            "session_id": self.session_id,
            "goal": user_goal,
            "model": self.model,
            "token_summary": result["token_summary"],
            "agents_spawned": result["agents_spawned"],
        }
        session_json_path = session_dir / "session.json"
        session_json_path.write_text(json.dumps(session_meta, indent=2), encoding="utf-8")

        # output.md — final synthesized output
        output_md_path = session_dir / "output.md"
        output_md_path.write_text(result["output"], encoding="utf-8")

        logger.info("Session output written to %s", session_dir)
