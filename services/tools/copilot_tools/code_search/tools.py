"""
Code search primitives: grep (ripgrep wrapper) and glob_search.

Both are designed for speed-first, low-overhead discovery inside a repository.
"""

from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputMode(str, Enum):
    FILES_WITH_MATCHES = "files_with_matches"
    CONTENT = "content"
    COUNT = "count"


# ---------------------------------------------------------------------------
# grep — ripgrep wrapper
# ---------------------------------------------------------------------------

def grep(
    pattern: str,
    *,
    path: Optional[str] = None,
    output_mode: str = "files_with_matches",
    glob_filter: Optional[str] = None,
    file_type: Optional[str] = None,
    case_insensitive: bool = False,
    line_numbers: bool = False,
    after_context: Optional[int] = None,
    before_context: Optional[int] = None,
    context: Optional[int] = None,
    multiline: bool = False,
    head_limit: Optional[int] = None,
) -> str:
    """Fast code search using **ripgrep** (``rg``).

    Parameters
    ----------
    pattern:
        Regex pattern to search for.
    path:
        File or directory to search.  Defaults to cwd.
    output_mode:
        ``"files_with_matches"`` | ``"content"`` | ``"count"``.
    glob_filter:
        Glob to filter files (e.g. ``"*.py"``).
    file_type:
        Ripgrep type filter (e.g. ``"py"``, ``"js"``).
    case_insensitive:
        Enable ``-i`` flag.
    line_numbers:
        Show line numbers (requires ``output_mode="content"``).
    after_context / before_context / context:
        Context lines around matches (requires ``output_mode="content"``).
    multiline:
        Allow patterns that span multiple lines.
    head_limit:
        Max number of results.

    Returns
    -------
    Ripgrep output as a string.
    """
    mode = OutputMode(output_mode)
    search_path = Path(path) if path else Path.cwd()

    args: list[str] = ["rg"]

    # Output mode flags
    if mode == OutputMode.FILES_WITH_MATCHES:
        args.append("--files-with-matches")
    elif mode == OutputMode.COUNT:
        args.append("--count")
    # content mode uses default rg output

    # Optional flags
    if glob_filter:
        args.extend(["--glob", glob_filter])
    if file_type:
        args.extend(["--type", file_type])
    if case_insensitive:
        args.append("--ignore-case")
    if line_numbers and mode == OutputMode.CONTENT:
        args.append("--line-number")
    if after_context is not None and mode == OutputMode.CONTENT:
        args.extend(["--after-context", str(after_context)])
    if before_context is not None and mode == OutputMode.CONTENT:
        args.extend(["--before-context", str(before_context)])
    if context is not None and mode == OutputMode.CONTENT:
        args.extend(["--context", str(context)])
    if multiline:
        args.append("--multiline")
    if head_limit is not None:
        args.extend(["--max-count", str(head_limit)])

    # Pattern + path
    args.append(pattern)
    args.append(str(search_path))

    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout


# ---------------------------------------------------------------------------
# glob_search — file pattern matching
# ---------------------------------------------------------------------------

def glob_search(
    pattern: str,
    *,
    path: Optional[str] = None,
) -> list[str]:
    """Find files by glob pattern.

    Parameters
    ----------
    pattern:
        Glob pattern (e.g. ``"**/*.py"``, ``"src/**/*.ts"``).
    path:
        Base directory.  Defaults to cwd.

    Returns
    -------
    List of matching absolute file paths.
    """
    base = Path(path) if path else Path.cwd()
    matches = sorted(base.glob(pattern))
    return [str(m.resolve()) for m in matches if not any(p.startswith(".") for p in m.parts)]
