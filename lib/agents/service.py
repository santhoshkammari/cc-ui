"""
Unified AI Agent Harness — AgentService

Central orchestrator: registers agents, manages sessions, routes messages.
Inspired by VS Code Copilot AgentService with _providers map + session routing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncIterator

from .base import BaseAgent
from .events import (
    AgentEvent,
    DeltaEvent,
    ErrorEvent,
    IdleEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    UsageEvent,
    ReasoningEvent,
)
from .session import (
    AgentSession,
    ResponsePart,
    SessionState,
    ToolCallState,
    Turn,
    UsageInfo,
    UserMessage,
)
from .tools import ToolRegistry, tool_registry

logger = logging.getLogger(__name__)


class AgentService:
    """
    Singleton service that manages all agents and their sessions.

    Usage:
        svc = AgentService()
        svc.register(MyCodingAgent())
        session = await svc.create_session("claude-code", model="sonnet")
        async for event in svc.send_message(session.id, "Fix the bug"):
            handle(event)
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._session_to_agent: dict[str, str] = {}  # session_id → agent_id
        self._default_agent: str | None = None
        self._lock = threading.Lock()
        self.tool_registry = registry or tool_registry

    # ── Agent Registration ──

    def register(self, agent: BaseAgent) -> None:
        """Register an agent. First registered becomes default."""
        with self._lock:
            self._agents[agent.id] = agent
            if self._default_agent is None:
                self._default_agent = agent.id
        logger.info(f"Registered agent: {agent.id} ({agent.name})")

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            self._agents.pop(agent_id, None)
            if self._default_agent == agent_id:
                self._default_agent = next(iter(self._agents), None)

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents with metadata."""
        return [a.to_dict() for a in self._agents.values()]

    @property
    def default_agent_id(self) -> str | None:
        return self._default_agent

    @default_agent_id.setter
    def default_agent_id(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._default_agent = agent_id

    # ── Session Management ──

    async def create_session(
        self,
        agent_id: str | None = None,
        model: str = "",
        cwd: str = "",
        config: dict[str, Any] | None = None,
    ) -> AgentSession:
        """Create a new session with the specified agent."""
        aid = agent_id or self._default_agent
        if aid is None:
            raise ValueError("No agents registered")

        agent = self._agents.get(aid)
        if agent is None:
            raise ValueError(f"Agent '{aid}' not found")

        session = await agent.create_session(model=model, cwd=cwd, config=config)
        session.agent_id = aid

        with self._lock:
            self._sessions[session.id] = session
            self._session_to_agent[session.id] = aid

        await agent.on_session_created(session)
        logger.info(f"Session {session.id} created for agent {aid}")
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, agent_id: str | None = None) -> list[AgentSession]:
        """List sessions, optionally filtered by agent."""
        sessions = list(self._sessions.values())
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        return sessions

    # ── Message Handling ──

    async def send_message(
        self,
        session_id: str,
        text: str,
        attachments: list[dict] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Send a message to a session and stream back events.
        Also updates session state (turns, usage) from the events.
        """
        session = self._sessions.get(session_id)
        if session is None:
            yield ErrorEvent(error_type="session_not_found", message=f"Session '{session_id}' not found")
            return

        agent_id = self._session_to_agent.get(session_id)
        agent = self._agents.get(agent_id) if agent_id else None
        if agent is None:
            yield ErrorEvent(error_type="agent_not_found", message=f"Agent for session not found")
            return

        # Build user message
        from .session import Attachment
        atts = []
        if attachments:
            for a in attachments:
                atts.append(Attachment(**a))
        user_msg = UserMessage(text=text, attachments=atts)

        # Start turn
        turn = session.start_turn(user_msg)

        try:
            async for event in agent.send_message(session, user_msg):
                # Update session state from events
                self._apply_event(turn, event)
                yield event

            # Turn completed successfully
            session.complete_turn()
            await agent.on_turn_complete(session)
            yield IdleEvent()

        except asyncio.CancelledError:
            session.cancel_turn()
            raise
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            session.fail_turn(error_msg)
            yield ErrorEvent(error_type="execution_error", message=error_msg)
            yield IdleEvent()

    def _apply_event(self, turn: Turn, event: AgentEvent) -> None:
        """Update turn state from an incoming event."""
        if isinstance(event, DeltaEvent):
            turn.add_text(event.content)

        elif isinstance(event, ReasoningEvent):
            turn.add_reasoning(event.content)

        elif isinstance(event, ToolStartEvent):
            tc = ToolCallState(
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                display_name=event.display_name,
                status="running",
                input_args=event.input_args,
            )
            turn.add_tool_call(tc)

        elif isinstance(event, ToolCompleteEvent):
            # Find and update the matching tool call
            for part in reversed(turn.response_parts):
                if part.tool_call and part.tool_call.tool_call_id == event.tool_call_id:
                    part.tool_call.status = "complete"
                    part.tool_call.result = event.result
                    part.tool_call.success = event.success
                    break

        elif isinstance(event, UsageEvent):
            if turn.usage is None:
                turn.usage = UsageInfo()
            turn.usage.input_tokens += event.input_tokens
            turn.usage.output_tokens += event.output_tokens
            turn.usage.cache_read_tokens += event.cache_read_tokens
            turn.usage.cache_write_tokens += event.cache_write_tokens
            turn.usage.total_cost_usd += event.total_cost_usd

        elif isinstance(event, ErrorEvent):
            if not event.recoverable:
                turn.fail(event.message)

    # ── Session Lifecycle ──

    async def stop_session(self, session_id: str) -> None:
        """Stop the active turn in a session."""
        session = self._sessions.get(session_id)
        if not session:
            return
        agent_id = self._session_to_agent.get(session_id)
        agent = self._agents.get(agent_id) if agent_id else None
        if agent:
            await agent.stop(session)
        session.cancel_turn()

    async def dispose_session(self, session_id: str) -> None:
        """Fully clean up and remove a session."""
        session = self._sessions.get(session_id)
        if not session:
            return
        agent_id = self._session_to_agent.get(session_id)
        agent = self._agents.get(agent_id) if agent_id else None
        if agent:
            await agent.dispose(session)
        session.finish()
        with self._lock:
            self._sessions.pop(session_id, None)
            self._session_to_agent.pop(session_id, None)

    # ── Shutdown ──

    async def shutdown(self) -> None:
        """Dispose all sessions and clean up."""
        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self.dispose_session(sid)
        logger.info("AgentService shut down")


# ─── Global instance ─────────────────────────────────────────────────

agent_service = AgentService()
