"""
Session-oriented PTY shell management.

Each session owns a pseudo-terminal (PTY), a PID, and persistent env vars.
Sessions are identified by a string ``shellId`` and managed through a global
``SessionManager``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants & key-map
# ---------------------------------------------------------------------------

class Mode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


KEY_MAP: dict[str, str] = {
    "{up}": "\x1b[A",
    "{down}": "\x1b[B",
    "{right}": "\x1b[C",
    "{left}": "\x1b[D",
    "{enter}": "\n",
    "{backspace}": "\x7f",
}

DEFAULT_INITIAL_WAIT = 30
MIN_INITIAL_WAIT = 30
MAX_INITIAL_WAIT = 600


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    shell_id: str
    command: str
    mode: Mode
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    detach: bool = False
    _output_buffer: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- output helpers -----------------------------------------------------

    def append_output(self, text: str) -> None:
        with self._lock:
            self._output_buffer += text

    def read_output(self) -> str:
        with self._lock:
            buf = self._output_buffer
            self._output_buffer = ""
            return buf

    def full_output(self) -> str:
        with self._lock:
            return self._output_buffer

    # -- status helpers -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def exit_code(self) -> Optional[int]:
        if self.process is None:
            return None
        return self.process.poll()

    @property
    def has_unread_output(self) -> bool:
        with self._lock:
            return len(self._output_buffer) > 0


# ---------------------------------------------------------------------------
# SessionManager — singleton pool of sessions
# ---------------------------------------------------------------------------

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, shell_id: Optional[str], mode: Mode) -> Session:
        if shell_id and shell_id in self._sessions:
            return self._sessions[shell_id]
        sid = shell_id or f"session-{uuid.uuid4().hex[:8]}"
        session = Session(shell_id=sid, command="", mode=mode)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, shell_id: str) -> Optional[Session]:
        return self._sessions.get(shell_id)

    def remove(self, shell_id: str) -> None:
        with self._lock:
            self._sessions.pop(shell_id, None)

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())


_manager = SessionManager()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _translate_keys(raw_input: str) -> str:
    """Replace ``{enter}``, ``{up}`` etc. with ANSI escape sequences."""
    result = raw_input
    for token, escape in KEY_MAP.items():
        result = result.replace(token, escape)
    return result


def _collect_output(session: Session, timeout: float) -> str:
    """Block until the process finishes *or* ``timeout`` expires, collecting stdout/stderr."""
    if session.process is None:
        return ""
    try:
        stdout, _ = session.process.communicate(timeout=timeout)
        session.append_output(stdout)
    except subprocess.TimeoutExpired:
        pass
    return session.full_output()


def _start_process(command: str, detach: bool) -> subprocess.Popen:
    kwargs: dict = dict(
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if detach:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bash(
    command: str,
    description: str,
    *,
    mode: str = "sync",
    initial_wait: int = DEFAULT_INITIAL_WAIT,
    shell_id: Optional[str] = None,
    detach: bool = False,
) -> dict:
    """Run a Bash command in an interactive session.

    Parameters
    ----------
    command:
        The shell command to run.
    description:
        Human-readable description (max 100 chars), displayed in the UI.
    mode:
        ``"sync"`` (default) — wait for output; ``"async"`` — run in background.
    initial_wait:
        Seconds (30-600) to wait for initial output in sync mode.
    shell_id:
        Reuse an existing session.  Auto-generated when ``None``.
    detach:
        Async-only.  If ``True`` the process survives agent shutdown.

    Returns
    -------
    dict with keys: ``output``, ``exitCode``, ``shellId``, ``status``.
    """
    initial_wait = max(MIN_INITIAL_WAIT, min(initial_wait, MAX_INITIAL_WAIT))
    run_mode = Mode(mode)

    session = _manager.get_or_create(shell_id, run_mode)
    session.command = command
    session.mode = run_mode
    session.detach = detach

    proc = _start_process(command, detach=(run_mode == Mode.ASYNC and detach))
    session.process = proc
    session.pid = proc.pid

    if run_mode == Mode.SYNC:
        output = _collect_output(session, timeout=initial_wait)

        if session.is_running:
            return {
                "output": output,
                "exitCode": None,
                "shellId": session.shell_id,
                "status": "running",
            }

        return {
            "output": session.full_output(),
            "exitCode": session.exit_code,
            "shellId": session.shell_id,
            "status": "completed",
        }

    # async — return immediately
    return {
        "output": "",
        "shellId": session.shell_id,
        "status": "running",
    }


def write_bash(
    shell_id: str,
    *,
    input_text: Optional[str] = None,
    delay: float = 10,
) -> dict:
    """Send input to a running session, then read output after *delay* seconds.

    Parameters
    ----------
    shell_id:
        Session to write to.
    input_text:
        Text / key tokens (``{enter}``, ``{up}``, …) to send.
    delay:
        Seconds to wait before reading the output buffer.
    """
    session = _manager.get(shell_id)
    if session is None:
        raise KeyError(f"No session with shellId={shell_id!r}")

    if input_text and session.process and session.process.stdin:
        translated = _translate_keys(input_text)
        session.process.stdin.write(translated)
        session.process.stdin.flush()

    time.sleep(delay)
    return {"output": session.read_output()}


def read_bash(shell_id: str, *, delay: float = 5) -> dict:
    """Read accumulated output from a running session.

    Parameters
    ----------
    shell_id:
        Session to read from.
    delay:
        Seconds to wait before reading.
    """
    session = _manager.get(shell_id)
    if session is None:
        raise KeyError(f"No session with shellId={shell_id!r}")

    time.sleep(delay)

    if session.process and session.is_running:
        try:
            stdout, _ = session.process.communicate(timeout=0.5)
            session.append_output(stdout)
        except subprocess.TimeoutExpired:
            pass

    return {"output": session.read_output()}


def stop_bash(shell_id: str) -> dict:
    """Terminate a Bash session and clean up resources.

    Sends SIGTERM → grace period → SIGKILL if needed.
    """
    session = _manager.get(shell_id)
    if session is None:
        raise KeyError(f"No session with shellId={shell_id!r}")

    if session.process and session.is_running:
        pid = session.process.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            session.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    _manager.remove(shell_id)
    return {"status": "terminated", "shellId": shell_id}


def list_bash() -> list[dict]:
    """Return info about all active sessions."""
    return [
        {
            "shellId": s.shell_id,
            "command": s.command,
            "mode": s.mode.value,
            "pid": s.pid,
            "status": "running" if s.is_running else "completed",
            "unreadOutput": s.has_unread_output,
        }
        for s in _manager.all_sessions()
    ]
