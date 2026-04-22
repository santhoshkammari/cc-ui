"""
Unified AI Agent Harness — Event Types

Inspired by VS Code Copilot's IAgentProgressEvent discriminated union.
All events are frozen dataclasses — immutable and serializable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> float:
    return time.time()


# ─── Event Types ──────────────────────────────────────────────────────

class EventType(str, Enum):
    DELTA = "delta"
    MESSAGE = "message"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    REASONING = "reasoning"
    USAGE = "usage"
    ERROR = "error"
    IDLE = "idle"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"


# ─── Individual Events ────────────────────────────────────────────────

@dataclass(frozen=True)
class DeltaEvent:
    """Streaming text chunk from the agent."""
    type: EventType = field(default=EventType.DELTA, init=False)
    content: str = ""
    message_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class MessageEvent:
    """Complete message (assistant or system)."""
    type: EventType = field(default=EventType.MESSAGE, init=False)
    role: str = "assistant"
    content: str = ""
    message_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ToolStartEvent:
    """Tool invocation started."""
    type: EventType = field(default=EventType.TOOL_START, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    display_name: str = ""
    input_args: str = ""
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ToolCompleteEvent:
    """Tool invocation finished."""
    type: EventType = field(default=EventType.TOOL_COMPLETE, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    success: bool = True
    result: str = ""
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ReasoningEvent:
    """Thinking/planning text from the agent."""
    type: EventType = field(default=EventType.REASONING, init=False)
    content: str = ""
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class UsageEvent:
    """Token usage and cost for a turn."""
    type: EventType = field(default=EventType.USAGE, init=False)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost_usd: float = 0.0
    session_id: str | None = None
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ErrorEvent:
    """Error during agent execution."""
    type: EventType = field(default=EventType.ERROR, init=False)
    error_type: str = "unknown"
    message: str = ""
    recoverable: bool = False
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class IdleEvent:
    """Turn complete, agent ready for next message."""
    type: EventType = field(default=EventType.IDLE, init=False)
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class SubagentStartEvent:
    """A sub-agent has been spawned."""
    type: EventType = field(default=EventType.SUBAGENT_START, init=False)
    agent_name: str = ""
    agent_id: str = ""
    parent_tool_call_id: str = ""
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class SubagentEndEvent:
    """A sub-agent has finished."""
    type: EventType = field(default=EventType.SUBAGENT_END, init=False)
    agent_id: str = ""
    success: bool = True
    timestamp: float = field(default_factory=_now)


# ─── Union type ───────────────────────────────────────────────────────

AgentEvent = Union[
    DeltaEvent,
    MessageEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ReasoningEvent,
    UsageEvent,
    ErrorEvent,
    IdleEvent,
    SubagentStartEvent,
    SubagentEndEvent,
]


# ─── Helpers ──────────────────────────────────────────────────────────

def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Serialize any AgentEvent to a JSON-safe dict."""
    from dataclasses import asdict
    d = asdict(event)
    d["type"] = event.type.value
    return d


def event_from_dict(d: dict[str, Any]) -> AgentEvent:
    """Deserialize a dict back to the correct AgentEvent type."""
    _MAP = {
        EventType.DELTA: DeltaEvent,
        EventType.MESSAGE: MessageEvent,
        EventType.TOOL_START: ToolStartEvent,
        EventType.TOOL_COMPLETE: ToolCompleteEvent,
        EventType.REASONING: ReasoningEvent,
        EventType.USAGE: UsageEvent,
        EventType.ERROR: ErrorEvent,
        EventType.IDLE: IdleEvent,
        EventType.SUBAGENT_START: SubagentStartEvent,
        EventType.SUBAGENT_END: SubagentEndEvent,
    }
    etype = EventType(d.pop("type"))
    cls = _MAP[etype]
    # Remove 'type' from kwargs since it's set by field default
    return cls(**d)
