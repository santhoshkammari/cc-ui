"""
CodingAgent — wraps existing agentic CLI providers (Claude Code, OpenCode).

These providers manage their own tools (file ops, bash, git) internally.
The agent simply bridges provider events → AgentEvent stream.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from lib.agents.base import BaseAgent
from lib.agents.events import (
    AgentEvent,
    DeltaEvent,
    ErrorEvent,
    ReasoningEvent,
    SubagentEndEvent,
    SubagentStartEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    UsageEvent,
)
from lib.agents.session import AgentSession, SessionState, UserMessage

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """
    Agentic coding assistant backed by CLI providers.
    Supports: Claude Code, OpenCode.
    Tools are managed by the underlying CLI — this agent just streams events.
    """

    def __init__(
        self,
        agent_id: str = "coding",
        agent_name: str = "Coding Agent",
        provider_name: str = "claude",
        default_mode: str = "bypassPermissions",
    ):
        self._id = agent_id
        self._name = agent_name
        self._provider_name = provider_name
        self._default_mode = default_mode
        self._active_providers: dict[str, Any] = {}  # session_id → provider instance

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Agentic coding assistant. Reads/writes files, runs commands, manages git, reasons through complex tasks."

    @property
    def icon(self) -> str:
        return "⚡"

    @property
    def models(self) -> list[dict[str, Any]]:
        try:
            from services.providers.registry import get_provider
            provider = get_provider(self._provider_name)
            return [
                {"id": m, "name": m, "provider": self._provider_name}
                for m in provider.available_models
            ]
        except Exception:
            return []

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_attachments(self) -> bool:
        return True

    @property
    def supports_subagents(self) -> bool:
        return True

    async def create_session(
        self,
        model: str = "",
        cwd: str = "",
        config: dict[str, Any] | None = None,
    ) -> AgentSession:
        session = AgentSession(
            model=model or (self.default_model or ""),
            cwd=cwd,
            config=config or {},
        )
        session.ready()
        return session

    async def send_message(
        self,
        session: AgentSession,
        message: UserMessage,
    ) -> AsyncIterator[AgentEvent]:
        from services.providers.base import EventType as PEventType, ProviderConfig
        from services.providers.registry import get_provider

        provider = get_provider(self._provider_name)
        self._active_providers[session.id] = provider

        config = ProviderConfig(
            model=session.model,
            mode=session.config.get("mode", self._default_mode),
            cwd=session.cwd,
            session_id=session.config.get("session_id"),
            extra=session.config.get("extra", {}),
        )

        history = session.build_history()

        try:
            async for event in provider.run(message.text, config, history):
                agent_event = self._map_event(event, PEventType)
                if agent_event is not None:
                    # Capture session_id from provider for resume support
                    if event.type == PEventType.COST:
                        sid = event.metadata.get("session_id")
                        if sid:
                            session.config["session_id"] = sid
                    yield agent_event

        except Exception as e:
            yield ErrorEvent(error_type="provider_error", message=str(e))
        finally:
            self._active_providers.pop(session.id, None)

    def _map_event(self, event: Any, PEventType: Any) -> AgentEvent | None:
        """Map a ProviderEvent to an AgentEvent."""
        if event.type == PEventType.TEXT:
            return DeltaEvent(content=event.content)

        elif event.type == PEventType.THINKING:
            return ReasoningEvent(content=event.content)

        elif event.type == PEventType.TOOL_START:
            return ToolStartEvent(
                tool_call_id=event.metadata.get("tool_call_id", ""),
                tool_name=event.metadata.get("title", "tool"),
                display_name=event.metadata.get("title", "⚙ Tool"),
                input_args=event.metadata.get("args", ""),
            )

        elif event.type == PEventType.TOOL_RESULT:
            return ToolCompleteEvent(
                tool_call_id=event.metadata.get("tool_call_id", ""),
                tool_name=event.metadata.get("title", "tool"),
                success=not event.metadata.get("is_error", False),
                result=event.content[:1000],
            )

        elif event.type == PEventType.COST:
            meta = event.metadata
            return UsageEvent(
                input_tokens=meta.get("usage", {}).get("input_tokens", 0),
                output_tokens=meta.get("usage", {}).get("output_tokens", 0),
                cache_read_tokens=meta.get("usage", {}).get("cache_read_input_tokens", 0),
                cache_write_tokens=meta.get("usage", {}).get("cache_creation_input_tokens", 0),
                total_cost_usd=meta.get("total_cost_usd", 0.0),
                session_id=meta.get("session_id"),
            )

        elif event.type == PEventType.AGENT_GROUP:
            status = event.metadata.get("status", "running")
            if status == "running":
                return SubagentStartEvent(
                    agent_name=event.metadata.get("agent_label", "Sub-agent"),
                    agent_id=event.metadata.get("agent_id", ""),
                )
            else:
                return SubagentEndEvent(
                    agent_id=event.metadata.get("agent_id", ""),
                    success=status != "error",
                )

        elif event.type == PEventType.ERROR:
            return ErrorEvent(error_type="provider_error", message=event.content)

        elif event.type == PEventType.DONE:
            return None  # Handled by send_message loop exit

        return None

    async def stop(self, session: AgentSession) -> None:
        provider = self._active_providers.get(session.id)
        if provider:
            await provider.stop()

    async def dispose(self, session: AgentSession) -> None:
        await self.stop(session)
        self._active_providers.pop(session.id, None)
