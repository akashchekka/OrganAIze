"""Spawner prompts — task assessment and decomposition."""

ASSESSMENT_PROMPT = """You are evaluating whether to decompose your current task into sub-tasks handled by specialist agents.

Task: "{task}"

Score on three axes (0.0 to 1.0):
- breadth: How many distinct skill domains does this require? (e.g., 0.2 = one skill, 0.9 = many different skills)
- depth: How specialized is the knowledge needed? (e.g., 0.2 = general, 0.9 = deep expert knowledge)
- parallelism: Can subtasks proceed independently? (e.g., 0.1 = very sequential, 0.9 = fully parallel)

Respond with ONLY valid JSON:
{{"breadth": 0.0, "depth": 0.0, "parallelism": 0.0, "reasoning": "..."}}"""

DECOMPOSITION_PROMPT = """You are an orchestrator agent decomposing a complex task into sub-tasks for specialist agents.

Task: "{task}"

Available agent roles: researcher, engineer, critic, qa, summarizer, pm, synthesizer

For each sub-agent, provide:
- name: A descriptive name (e.g., "ResearchAgent-FloorPlanModels")
- role: One of the available roles
- persona: One-line personality description
- expertise: List of skill keywords
- task: Clear task description for this agent
- success_criteria: How to know the sub-task is done
- output_format: "json" | "markdown" | "code" | "structured_report"
- tools_needed: List of tool names this agent needs
- needs_spawn: Whether this agent might need to spawn its own sub-agents (true/false)

Respond with ONLY valid JSON:
{{"agents": [...]}}"""
