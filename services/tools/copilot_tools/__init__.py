"""
copilot_tools — Python implementations of GitHub Copilot CLI tools.

Modules:
    bash_execution      — Shell session management (bash, write_bash, read_bash, stop_bash, list_bash)
    file_operations     — File I/O primitives (view, create, edit)
    code_search         — Code & file discovery (grep, glob)
    session_workflow    — Session state & user interaction (report_intent, ask_user, sql, fetch_docs)
    agent_orchestration — Multi-agent orchestration (task, read_agent, list_agents, skill)
    markdown            — Structured extraction from markdown documents
    web                 — Web search, fetch, and ChromaDB-backed content store
"""

from copilot_tools.bash_execution import (
    bash,
    write_bash,
    read_bash,
    stop_bash,
    list_bash,
)
from copilot_tools.file_operations import view, create, edit
from copilot_tools.code_search import grep, glob_search
from copilot_tools.session_workflow import (
    report_intent,
    ask_user,
    sql,
    fetch_copilot_cli_documentation,
)
from copilot_tools.agent_orchestration import (
    launch_task,
    read_agent,
    list_agents,
    execute_skill,
)
from copilot_tools.markdown import (
    get_overview as markdown_get_overview,
    get_headers as markdown_get_headers,
    get_section as markdown_get_section,
    get_intro as markdown_get_intro,
    get_links as markdown_get_links,
    get_tables_metadata as markdown_get_tables_metadata,
    get_table as markdown_get_table,
)
from copilot_tools.web import (
    web_fetch,
    web_search,
    web_store_get,
    web_store_get_text,
    web_store_search,
    web_store_list,
)

__all__ = [
    # bash_execution
    "bash",
    "write_bash",
    "read_bash",
    "stop_bash",
    "list_bash",
    # file_operations
    "view",
    "create",
    "edit",
    # code_search
    "grep",
    "glob_search",
    # session_workflow
    "report_intent",
    "ask_user",
    "sql",
    "fetch_copilot_cli_documentation",
    # agent_orchestration
    "launch_task",
    "read_agent",
    "list_agents",
    "execute_skill",
    # markdown
    "markdown_get_overview",
    "markdown_get_headers",
    "markdown_get_section",
    "markdown_get_intro",
    "markdown_get_links",
    "markdown_get_tables_metadata",
    "markdown_get_table",
    # web
    "web_search",
    "web_fetch",
    "web_store_get",
    "web_store_get_text",
    "web_store_search",
    "web_store_list",
]
