"""
Bash Execution Tools — interactive shell session management.

Provides five tools that give a full interactive shell:
    bash        — Run a command (sync or async), create/reuse sessions
    write_bash  — Send input (text + key sequences) to a running session
    read_bash   — Read accumulated output from a session
    stop_bash   — Terminate a session and clean up resources
    list_bash   — List all active sessions
"""

from copilot_tools.bash_execution.sessions import (
    bash,
    write_bash,
    read_bash,
    stop_bash,
    list_bash,
)

__all__ = ["bash", "write_bash", "read_bash", "stop_bash", "list_bash"]
