"""
Unified AI Agent Harness — Session & Turn Models

Inspired by VS Code Copilot's ITurn / IActiveTurn / session lifecycle.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ─── Enums ────────────────────────────────────────────────────────────

class SessionState(str, Enum):
    CREATING = "creating"
    READY = "ready"
    ACTIVE = "active"      # Currently processing a turn
    IDLE = "idle"          # Turn complete, waiting for input
    COMPLETED = "completed"
    ERROR = "error"


class TurnState(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# ─── Data Models ──────────────────────────────────────────────────────

@dataclass
class Attachment:
    """File or content attachment on a message."""
    type: str = "file"           # file | selection | directory | image
    path: str = ""
    display_name: str = ""
    content: str | None = None   # inline text for selections
    mime_type: str | None = None

    def to_dict(self) -> dict:
        d = {"type": self.type, "path": self.path, "display_name": self.display_name}
        if self.content is not None:
            d["content"] = self.content
        if self.mime_type is not None:
            d["mime_type"] = self.mime_type
        return d


@dataclass
class UserMessage:
    """A user's input in a turn."""
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "attachments": [a.to_dict() for a in self.attachments],
        }


@dataclass
class ToolCallState:
    """Tracks the lifecycle of a single tool call within a turn."""
    tool_call_id: str = ""
    tool_name: str = ""
    display_name: str = ""
    status: str = "pending"    # pending | running | complete | cancelled
    input_args: str = ""
    result: str | None = None
    success: bool | None = None

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "display_name": self.display_name,
            "status": self.status,
            "input_args": self.input_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class ResponsePart:
    """A piece of the agent's response (text, reasoning, tool call)."""
    kind: str = "markdown"   # markdown | reasoning | tool_call
    content: str = ""
    tool_call: ToolCallState | None = None

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "content": self.content}
        if self.tool_call:
            d["tool_call"] = self.tool_call.to_dict()
        return d


@dataclass
class UsageInfo:
    """Token usage for a turn."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def merge(self, other: UsageInfo) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.total_cost_usd += other.total_cost_usd


# ─── Turn ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single user→agent exchange."""
    id: str = field(default_factory=_uid)
    user_message: UserMessage = field(default_factory=UserMessage)
    response_parts: list[ResponsePart] = field(default_factory=list)
    usage: UsageInfo | None = None
    state: TurnState = TurnState.ACTIVE
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_text(self, content: str) -> None:
        """Append or extend the last markdown response part."""
        if self.response_parts and self.response_parts[-1].kind == "markdown":
            self.response_parts[-1].content += content
        else:
            self.response_parts.append(ResponsePart(kind="markdown", content=content))

    def add_reasoning(self, content: str) -> None:
        self.response_parts.append(ResponsePart(kind="reasoning", content=content))

    def add_tool_call(self, tc: ToolCallState) -> None:
        self.response_parts.append(ResponsePart(kind="tool_call", tool_call=tc))

    def complete(self) -> None:
        self.state = TurnState.COMPLETE
        self.finished_at = time.time()

    def fail(self, error: str) -> None:
        self.state = TurnState.ERROR
        self.error = error
        self.finished_at = time.time()

    def cancel(self) -> None:
        self.state = TurnState.CANCELLED
        self.finished_at = time.time()

    @property
    def answer(self) -> str:
        """Concatenated text from all markdown response parts."""
        return "".join(p.content for p in self.response_parts if p.kind == "markdown")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_message": self.user_message.to_dict(),
            "response_parts": [p.to_dict() for p in self.response_parts],
            "usage": self.usage.to_dict() if self.usage else None,
            "state": self.state.value,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ─── Session ──────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    """
    A conversation session bound to one agent.
    Tracks turns, state, and configuration.
    """
    id: str = field(default_factory=_uid)
    agent_id: str = ""
    model: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    active_turn: Turn | None = None
    state: SessionState = SessionState.CREATING
    cwd: str = ""
    branch: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    # ── Lifecycle ──

    def ready(self) -> None:
        self.state = SessionState.READY

    def start_turn(self, message: UserMessage) -> Turn:
        """Begin a new turn. Returns the active Turn."""
        turn = Turn(user_message=message)
        self.active_turn = turn
        self.state = SessionState.ACTIVE
        return turn

    def complete_turn(self) -> Turn | None:
        """Finalize the active turn and move to idle."""
        if self.active_turn is None:
            return None
        self.active_turn.complete()
        self.turns.append(self.active_turn)
        turn = self.active_turn
        self.active_turn = None
        self.state = SessionState.IDLE
        return turn

    def fail_turn(self, error: str) -> None:
        if self.active_turn:
            self.active_turn.fail(error)
            self.turns.append(self.active_turn)
            self.active_turn = None
        self.state = SessionState.IDLE

    def cancel_turn(self) -> None:
        if self.active_turn:
            self.active_turn.cancel()
            self.turns.append(self.active_turn)
            self.active_turn = None
        self.state = SessionState.IDLE

    def finish(self) -> None:
        self.state = SessionState.COMPLETED
        self.finished_at = time.time()

    # ── History helpers ──

    def build_history(self) -> list[dict[str, Any]]:
        """Build OpenAI-style message history from all completed turns."""
        messages: list[dict[str, Any]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user_message.text})
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
        return messages

    @property
    def total_usage(self) -> UsageInfo:
        """Aggregate usage across all turns."""
        total = UsageInfo()
        for turn in self.turns:
            if turn.usage:
                total.merge(turn.usage)
        if self.active_turn and self.active_turn.usage:
            total.merge(self.active_turn.usage)
        return total

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "model": self.model,
            "config": self.config,
            "turns": [t.to_dict() for t in self.turns],
            "active_turn": self.active_turn.to_dict() if self.active_turn else None,
            "state": self.state.value,
            "cwd": self.cwd,
            "branch": self.branch,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }
