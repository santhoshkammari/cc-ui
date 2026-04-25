"""
Session & Workflow Tools — state, user interaction, and self-documentation.

    report_intent                — Update the UI status bar with current activity
    ask_user                     — Pause and collect user input (choices or freeform)
    sql                          — Execute SQLite queries against a session-scoped DB
    fetch_copilot_cli_documentation — Load bundled CLI docs for introspection
"""

from copilot_tools.session_workflow.tools import (
    report_intent,
    ask_user,
    sql,
    fetch_copilot_cli_documentation,
)

__all__ = ["report_intent", "ask_user", "sql", "fetch_copilot_cli_documentation"]
