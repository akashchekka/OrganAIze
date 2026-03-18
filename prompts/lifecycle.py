"""Lifecycle prompts — resurrection context."""

RESURRECTION_CONTEXT_TEMPLATE = """
--- RESURRECTION CONTEXT ---
You were previously working on a task and were paused.
Your progress so far: {partial_results}
Last step completed: {last_step}
{resume_instruction}
--- END RESURRECTION CONTEXT ---
"""
