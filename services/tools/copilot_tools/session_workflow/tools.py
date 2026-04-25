"""
Session workflow primitives: report_intent, ask_user, sql, fetch_copilot_cli_documentation.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Session state (singleton per process)
# ---------------------------------------------------------------------------

@dataclass
class _SessionState:
    current_intent: str = ""
    intent_history: list[dict] = field(default_factory=list)
    db_connection: Optional[sqlite3.Connection] = None
    db_path: Optional[Path] = None

    def get_db(self) -> sqlite3.Connection:
        if self.db_connection is None:
            self.db_connection = sqlite3.connect(":memory:")
            self.db_connection.row_factory = sqlite3.Row
            self._bootstrap_schema()
        return self.db_connection

    def _bootstrap_schema(self) -> None:
        db = self.db_connection
        assert db is not None
        db.executescript(
            """\
            CREATE TABLE IF NOT EXISTS todos (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT,
                status      TEXT DEFAULT 'pending',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS todo_deps (
                todo_id    TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                PRIMARY KEY (todo_id, depends_on),
                FOREIGN KEY (todo_id)    REFERENCES todos(id),
                FOREIGN KEY (depends_on) REFERENCES todos(id)
            );
            """
        )
        db.commit()


_state = _SessionState()


# ---------------------------------------------------------------------------
# report_intent
# ---------------------------------------------------------------------------

def report_intent(intent: str) -> str:
    """Update the current activity shown in the UI status bar.

    Parameters
    ----------
    intent:
        Short description, **4 words max**, gerund form.
        e.g. ``"Exploring codebase"``, ``"Fixing auth middleware"``.

    Returns
    -------
    ``"Intent logged"``
    """
    words = intent.strip().split()
    if len(words) > 4:
        raise ValueError(f"Intent must be ≤ 4 words, got {len(words)}: {intent!r}")

    _state.current_intent = intent
    _state.intent_history.append({"timestamp": time.time(), "intent": intent})
    return "Intent logged"


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------

def ask_user(
    question: str,
    *,
    choices: Optional[list[str]] = None,
    allow_freeform: bool = True,
) -> str:
    """Pause execution and ask the user a question.

    Parameters
    ----------
    question:
        Single, clear question.
    choices:
        Optional list of multiple-choice options.
    allow_freeform:
        Allow free-text input alongside choices.

    Returns
    -------
    The user's answer as a string.
    """
    print(f"\n{'─' * 60}")
    print(f"  ❓  {question}")

    if choices:
        for i, choice in enumerate(choices, start=1):
            print(f"     {i}. {choice}")
        if allow_freeform:
            print("     (or type your own answer)")

    print(f"{'─' * 60}")
    answer = input("  ▸ Your answer: ").strip()

    # If user typed a number and choices exist, resolve it
    if choices and answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    return answer


# ---------------------------------------------------------------------------
# sql
# ---------------------------------------------------------------------------

def sql(query: str, description: str) -> dict:
    """Execute a SQLite query against the session-scoped database.

    Parameters
    ----------
    query:
        Any SQLite-compatible SQL statement.
    description:
        2-5 word summary for logging (e.g. ``"Insert auth todos"``).

    Returns
    -------
    dict with ``rows`` (for SELECT) and ``changes`` (for INSERT/UPDATE/DELETE).
    """
    db = _state.get_db()
    cursor = db.execute(query)
    db.commit()

    if cursor.description:
        rows = [dict(row) for row in cursor.fetchall()]
        return {"rows": rows, "changes": 0, "description": description}

    return {"rows": [], "changes": cursor.rowcount, "description": description}


# ---------------------------------------------------------------------------
# fetch_copilot_cli_documentation
# ---------------------------------------------------------------------------

_DOCS_SEARCH_DIRS = [
    Path.home() / ".copilot",
    Path("/usr/share/copilot-cli"),
    Path(__file__).resolve().parent.parent.parent,  # fallback: package root
]


def fetch_copilot_cli_documentation() -> dict:
    """Load bundled Copilot CLI documentation.

    Searches known locations for ``README.md`` and ``help.txt``.

    Returns
    -------
    dict with ``readme`` and ``help_output`` strings.
    """
    readme_text = ""
    help_text = ""

    for base in _DOCS_SEARCH_DIRS:
        readme_candidate = base / "README.md"
        help_candidate = base / "help.txt"

        if not readme_text and readme_candidate.is_file():
            readme_text = readme_candidate.read_text(errors="replace")
        if not help_text and help_candidate.is_file():
            help_text = help_candidate.read_text(errors="replace")

        if readme_text and help_text:
            break

    return {
        "readme": readme_text or "[README.md not found in known locations]",
        "help_output": help_text or "[help.txt not found in known locations]",
    }
