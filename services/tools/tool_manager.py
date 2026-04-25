"""
Tool manager — OpenAI-format tool schemas + executor for local LLM tool calling.

Wraps copilot_tools functions into OpenAI-compatible tool definitions and provides
a unified execution interface for the vLLM agentic loop.
"""
from __future__ import annotations

import json
import traceback
from typing import Any

# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions (subset safe for local LLMs)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return output. Use for executing programs, installing packages, running scripts, checking system state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "description": {"type": "string", "description": "Brief description of what this command does (max 100 chars)"},
                },
                "required": ["command", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "Read a file's content (with line numbers) or list a directory's contents. For large files, use view_range to read specific line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to file or directory"},
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start_line, end_line] to read a range. Use [-1] for end of file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": "Create a new file with the given content. Parent directory must exist. File must NOT already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path for the new file"},
                    "file_text": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "file_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exactly one occurrence of old_str with new_str in a file. old_str must match exactly and be unique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to edit"},
                    "old_str": {"type": "string", "description": "The exact string to find and replace (must be unique in file)"},
                    "new_str": {"type": "string", "description": "The replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using ripgrep. Returns matching files or matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in (default: current dir)"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": "Output format: files_with_matches (default), content (show lines), count",
                    },
                    "glob_filter": {"type": "string", "description": "Glob pattern to filter files (e.g. '*.py')"},
                    "case_insensitive": {"type": "boolean", "description": "Case insensitive search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_search",
            "description": "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match files"},
                    "path": {"type": "string", "description": "Base directory to search from"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return summarized results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 4)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its content as markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _get_tool_functions() -> dict[str, Any]:
    """Lazy-load tool functions from copilot_tools."""
    import sys, os
    # Ensure our tools dir is on sys.path
    tools_dir = os.path.join(os.path.dirname(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from copilot_tools.bash_execution import bash
    from copilot_tools.file_operations import view, create, edit
    from copilot_tools.code_search import grep, glob_search

    fns: dict[str, Any] = {
        "bash": bash,
        "view": view,
        "create": create,
        "edit": edit,
        "grep": grep,
        "glob_search": glob_search,
    }

    # Web tools may fail if deps missing — graceful fallback
    try:
        from copilot_tools.web import web_search, web_fetch
        fns["web_search"] = web_search
        fns["web_fetch"] = web_fetch
    except ImportError:
        pass

    return fns


_cached_fns: dict[str, Any] | None = None


def execute_tool(name: str, arguments: dict, cwd: str = "") -> tuple[str, bool]:
    """Execute a tool by name with given arguments.

    Returns (result_text, is_error).
    """
    global _cached_fns
    if _cached_fns is None:
        _cached_fns = _get_tool_functions()

    fn = _cached_fns.get(name)
    if fn is None:
        return f"Unknown tool: {name}", True

    try:
        # Inject cwd for tools that need it
        if name == "bash" and cwd:
            cmd = arguments.get("command", "")
            if not cmd.startswith("cd "):
                arguments["command"] = f"cd {cwd} && {cmd}"
        elif name in ("grep", "glob_search") and cwd and "path" not in arguments:
            arguments["path"] = cwd

        result = fn(**arguments)

        # Normalize result to string
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str), False
        elif isinstance(result, list):
            return json.dumps(result, indent=2, default=str), False
        else:
            return str(result), False

    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()[-500:]}", True


def get_system_prompt(cwd: str = "") -> str:
    """Build a system prompt for the local LLM with tool-use instructions."""
    return f"""You are an expert AI coding assistant. You have access to tools to help you complete tasks.

Current working directory: {cwd or '~/'}

When you need to perform actions, use the available tools. Always think step by step:
1. Understand what the user wants
2. Plan which tools to use
3. Execute tools one at a time
4. Report results clearly

Be concise and precise. When editing files, read them first to understand the content."""
