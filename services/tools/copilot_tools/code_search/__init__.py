"""
Code Search Tools — fast content and filename searching.

    grep        — Search file *contents* via regex (wraps ripgrep)
    glob_search — Search file *names* via glob patterns
"""

from copilot_tools.code_search.tools import grep, glob_search

__all__ = ["grep", "glob_search"]
