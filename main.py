"""OrganAIze — Self-Organizing Agentic AI

Usage:
    python main.py "Build an app that converts floor plans into realistic images"
    python main.py --max-tokens 100000 --model gpt-4o "Your goal here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from core.genesis import GenesisAgent


def setup_logging(verbose: bool = False) -> None:
    """Configure Python logging for OrganAIze."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def print_summary(result: dict) -> None:
    """Print session results and token summary."""
    print("\n" + "=" * 70)
    print("ORGANAIZE SESSION COMPLETE")
    print("=" * 70)

    print(f"\nSession ID: {result['session_id']}")
    print(f"Agents spawned: {result['agents_spawned']}")

    summary = result["token_summary"]
    print(f"\n--- Token Summary ---")
    print(f"Total input tokens:  {summary['total_input_tokens']:,}")
    print(f"Total output tokens: {summary['total_output_tokens']:,}")
    print(f"Total tokens:        {summary['total_tokens']:,}")
    print(f"Total LLM calls:     {summary['total_llm_calls']}")
    print(f"Tokens remaining:    {summary['tokens_remaining']:,}")

    print(f"\n--- Final Output ---")
    print(result["output"][:5000])
    if len(result.get("output", "")) > 5000:
        print("\n...[output truncated]")


async def async_main(goal: str, max_tokens: int, model: str | None, verbose: bool) -> None:
    setup_logging(verbose=verbose)
    from config import DEFAULT_LLM_MODEL
    model = model or DEFAULT_LLM_MODEL
    print(f"OrganAIze - Starting session")
    print(f"   Goal:       {goal}")
    print(f"   Max tokens: {max_tokens:,}")
    print(f"   Model:      {model}")
    print()

    genesis = GenesisAgent(max_session_tokens=max_tokens, model=model)
    result = await genesis.run(goal)
    print_summary(result)


def main():
    parser = argparse.ArgumentParser(
        description="OrganAIze — Self-Organizing Agentic AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("goal", help="The user goal to achieve")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500000,
        help="Max total tokens for the session (default: 500000)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for the genesis agent (default: from .env or azure/gpt-4o)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args.goal, args.max_tokens, args.model, args.verbose))


if __name__ == "__main__":
    main()
