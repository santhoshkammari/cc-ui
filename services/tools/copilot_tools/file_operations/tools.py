"""
Filesystem primitives: view, create, edit.

All paths must be absolute (enforced via ``pathlib.Path``).
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 50 * 1024  # 50 KB
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".webp", ".bmp", ".ico", ".tiff", ".tif",
}
DIR_LIST_DEPTH = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_absolute(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError(f"Path MUST be absolute, got: {path}")


def _add_line_numbers(content: str) -> str:
    lines = content.splitlines(keepends=True)
    numbered = []
    for i, line in enumerate(lines, start=1):
        numbered.append(f"{i}. {line.rstrip()}")
    return "\n".join(numbered)


def _list_directory(path: Path, *, depth: int = DIR_LIST_DEPTH, _current: int = 0) -> str:
    """Recursively list non-hidden entries up to *depth* levels."""
    entries: list[str] = []
    indent = "  " * _current

    try:
        children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"{indent}{path.name}/ [permission denied]"

    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(f"{indent}{child.name}/")
            if _current < depth - 1:
                entries.append(_list_directory(child, depth=depth, _current=_current + 1))
        else:
            entries.append(f"{indent}{child.name}")

    return "\n".join(entries)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def view(
    path: str,
    *,
    view_range: Optional[tuple[int, int]] = None,
    force_read_large_files: bool = False,
) -> str | dict:
    """View a file, directory, or image.

    Parameters
    ----------
    path:
        Absolute path to file or directory.  Must exist.
    view_range:
        ``(start, end)`` — 1-indexed line range.  ``end=-1`` means EOF.
    force_read_large_files:
        Skip the 50 KB truncation guard.

    Returns
    -------
    * **str** with line-numbered content for regular files.
    * **dict** ``{"base64": ..., "mimeType": ...}`` for images.
    * **str** directory listing for directories.
    """
    p = Path(path)
    _ensure_absolute(p)

    if not p.exists():
        raise FileNotFoundError(f"File MUST exist to view: {p}")

    # --- Directory ---
    if p.is_dir():
        return _list_directory(p)

    # --- Image ---
    if _is_image(p):
        data = p.read_bytes()
        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        return {"base64": base64.b64encode(data).decode(), "mimeType": mime}

    # --- Regular file ---
    size = p.stat().st_size
    if size > MAX_FILE_SIZE and not force_read_large_files:
        content = p.read_text(errors="replace")[:MAX_FILE_SIZE]
        return (
            _add_line_numbers(content)
            + "\n[File truncated at 50KB. Use view_range to read specific sections.]"
        )

    content = p.read_text(errors="replace")

    if view_range is not None:
        start, end = view_range
        lines = content.splitlines()
        if end == -1:
            end = len(lines)
        content = "\n".join(lines[start - 1 : end])

    return _add_line_numbers(content)


def create(path: str, file_text: str) -> dict:
    """Create a new file.  Refuses if the file already exists.

    Parameters
    ----------
    path:
        Absolute path.  File MUST NOT exist.
    file_text:
        Complete content to write.
    """
    p = Path(path)
    _ensure_absolute(p)

    if p.exists():
        raise FileExistsError(
            f"File already exists at {p}. Cannot overwrite. Use edit() for existing files."
        )

    if not p.parent.exists():
        raise FileNotFoundError(
            f"Parent directory does not exist: {p.parent}. Create it first."
        )

    p.write_text(file_text)
    return {"success": True, "path": str(p), "bytes_written": len(file_text)}


def edit(path: str, old_str: str, new_str: str) -> dict:
    """Replace exactly one occurrence of *old_str* with *new_str*.

    Parameters
    ----------
    path:
        Absolute path.  File MUST exist.
    old_str:
        Exact text to find (must appear exactly once).
    new_str:
        Replacement text.
    """
    p = Path(path)
    _ensure_absolute(p)

    if not p.exists():
        raise FileNotFoundError(
            f"File MUST exist to edit: {p}. Use create() for new files."
        )

    content = p.read_text(errors="replace")
    occurrences = content.count(old_str)

    if occurrences == 0:
        raise ValueError(
            f"old_str not found in {p}. "
            "Ensure the string matches exactly, including whitespace."
        )
    if occurrences > 1:
        raise ValueError(
            f"old_str appears {occurrences} times in {p}. "
            "Not unique — include more surrounding context."
        )

    new_content = content.replace(old_str, new_str, 1)
    p.write_text(new_content)
    return {"success": True, "path": str(p)}
