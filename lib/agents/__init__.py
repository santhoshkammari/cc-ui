"""
Unified AI Agent Harness

Public API:
    from lib.agents import (
        # Core
        BaseAgent, AgentService, agent_service,
        # Events
        AgentEvent, DeltaEvent, MessageEvent, ToolStartEvent, ToolCompleteEvent,
        ReasoningEvent, UsageEvent, ErrorEvent, IdleEvent,
        # Session
        AgentSession, Turn, UserMessage, Attachment,
        # Tools
        ToolRegistry, ToolDefinition, ToolResult, tool_registry, fn_to_tool,
    )
"""

from .events import (
    EventType,
    AgentEvent,
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
    event_to_dict,
    event_from_dict,
)

from .session import (
    SessionState,
    TurnState,
    Attachment,
    UserMessage,
    ToolCallState,
    ResponsePart,
    UsageInfo,
    Turn,
    AgentSession,
)

from .base import BaseAgent

from .tools import (
    ToolDefinition,
    ToolResult,
    ToolRegistry,
    tool_registry,
    fn_to_tool,
)

from .service import (
    AgentService,
    agent_service,
)

__all__ = [
    # Events
    "EventType",
    "AgentEvent",
    "DeltaEvent",
    "MessageEvent",
    "ToolStartEvent",
    "ToolCompleteEvent",
    "ReasoningEvent",
    "UsageEvent",
    "ErrorEvent",
    "IdleEvent",
    "SubagentStartEvent",
    "SubagentEndEvent",
    "event_to_dict",
    "event_from_dict",
    # Session
    "SessionState",
    "TurnState",
    "Attachment",
    "UserMessage",
    "ToolCallState",
    "ResponsePart",
    "UsageInfo",
    "Turn",
    "AgentSession",
    # Agent
    "BaseAgent",
    # Tools
    "ToolDefinition",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "fn_to_tool",
    # Service
    "AgentService",
    "agent_service",
]
