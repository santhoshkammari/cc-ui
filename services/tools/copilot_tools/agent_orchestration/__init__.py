"""
Agent Orchestration Tools — multi-agent management.

    launch_task   — Spawn a specialised sub-agent (explore, task, general-purpose, code-review)
    read_agent    — Retrieve status & results from a background agent
    list_agents   — List all active / completed agents
    execute_skill — Execute a named skill in the main conversation context
"""

from copilot_tools.agent_orchestration.tools import (
    launch_task,
    read_agent,
    list_agents,
    execute_skill,
)

__all__ = ["launch_task", "read_agent", "list_agents", "execute_skill"]
