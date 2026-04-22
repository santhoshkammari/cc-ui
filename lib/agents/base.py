"""
Unified AI Agent Harness — BaseAgent Abstract Class

Inspired by VS Code Copilot's IAgent interface.
Every agent (coding, chat, custom) implements this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from .events import AgentEvent
from .session import AgentSession, UserMessage


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Lifecycle:
        1. Register with AgentService
        2. create_session() → AgentSession
        3. send_message() → AsyncIterator[AgentEvent]  (streaming)
        4. stop() / dispose()

    Subclasses must implement: create_session, send_message, stop.
    """

    # ── Identity (set by subclass) ──

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique agent identifier (e.g. 'claude-code', 'chat', 'inhouse')."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""
        ...

    @property
    def description(self) -> str:
        """Short description of what this agent does."""
        return ""

    @property
    def icon(self) -> str:
        """Emoji or icon for UI display."""
        return "🤖"

    # ── Models ──

    @property
    def models(self) -> list[dict[str, Any]]:
        """
        Available models for this agent.
        Each dict: {"id": str, "name": str, "provider": str, ...}
        Override to list models dynamically.
        """
        return []

    @property
    def default_model(self) -> str | None:
        """Default model ID. None means agent picks internally."""
        models = self.models
        return models[0]["id"] if models else None

    # ── Tools ──

    @property
    def tool_tags(self) -> list[str]:
        """
        Tags of tools this agent wants from the ToolRegistry.
        E.g. ["file", "code", "bash"] → agent gets file ops, grep, bash tools.
        Override to opt-in to specific tool categories.
        Return empty list if agent manages its own tools.
        """
        return []

    @property
    def tool_names(self) -> list[str] | None:
        """
        Explicit list of tool names this agent wants.
        Takes priority over tool_tags if set. None = use tags.
        """
        return None

    # ── Capabilities ──

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def supports_attachments(self) -> bool:
        return False

    @property
    def supports_subagents(self) -> bool:
        return False

    # ── Lifecycle ──

    @abstractmethod
    async def create_session(
        self,
        model: str = "",
        cwd: str = "",
        config: dict[str, Any] | None = None,
    ) -> AgentSession:
        """
        Create and return a new session for this agent.
        The session is in READY state after this returns.
        """
        ...

    @abstractmethod
    async def send_message(
        self,
        session: AgentSession,
        message: UserMessage,
    ) -> AsyncIterator[AgentEvent]:
        """
        Send a message to the agent and stream back events.
        The caller is responsible for updating session state from events.
        This must be an async generator that yields AgentEvent instances.
        """
        ...
        # Type hint says AsyncIterator but implementation should be `async def ... yield`
        yield  # type: ignore  # make this a generator

    @abstractmethod
    async def stop(self, session: AgentSession) -> None:
        """Stop the active turn in the given session."""
        ...

    async def dispose(self, session: AgentSession) -> None:
        """
        Clean up all resources for a session.
        Default: just stop. Override for custom cleanup.
        """
        await self.stop(session)

    # ── Optional hooks ──

    async def on_session_created(self, session: AgentSession) -> None:
        """Hook called after session creation. Override for setup."""
        pass

    async def on_turn_complete(self, session: AgentSession) -> None:
        """Hook called after a turn completes. Override for cleanup."""
        pass

    # ── Repr ──

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id!r} name={self.name!r}>"

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent metadata for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "models": self.models,
            "default_model": self.default_model,
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "supports_attachments": self.supports_attachments,
            "supports_subagents": self.supports_subagents,
        }
