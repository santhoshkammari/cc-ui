"""
InHouseAgent — wraps the existing AIAgent from lab/src/ai/ai.py.

Bridges the ai.py event model (Text, ToolCall, ToolResult, StepResult)
to the unified AgentEvent stream.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from lib.agents.base import BaseAgent
from lib.agents.events import (
    AgentEvent,
    DeltaEvent,
    ErrorEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    UsageEvent,
)
from lib.agents.session import AgentSession, UserMessage

logger = logging.getLogger(__name__)


class InHouseAgent(BaseAgent):
    """
    Agent wrapping the in-house AIAgent framework (ai.py).
    Supports tool use via fn_to_tool, multiple LLM backends.
    """

    def __init__(
        self,
        agent_id: str = "inhouse",
        agent_name: str = "In-House AI",
        default_model: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list | None = None,
    ):
        self._id = agent_id
        self._name = agent_name
        self._default_model = default_model
        self._base_url = base_url
        self._api_key = api_key
        self._tools = tools

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "In-house AI agent with custom tool support."

    @property
    def icon(self) -> str:
        return "🧪"

    @property
    def default_model(self) -> str | None:
        return self._default_model or None

    @property
    def supports_tools(self) -> bool:
        return True

    async def create_session(
        self,
        model: str = "",
        cwd: str = "",
        config: dict[str, Any] | None = None,
    ) -> AgentSession:
        session = AgentSession(
            model=model or self._default_model,
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
        import os
        import sys

        # Add lab/src to path so we can import ai module
        lab_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lab", "src")
        lab_src = os.path.normpath(lab_src)
        if lab_src not in sys.path:
            sys.path.insert(0, lab_src)

        try:
            from ai.ai import AIAgent, AIConfig, Chat, Text, ToolCall, ToolResult, StepResult
        except ImportError as e:
            yield ErrorEvent(error_type="import_error", message=f"Cannot import AIAgent: {e}")
            return

        api_key = self._api_key or session.config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = self._base_url or session.config.get("base_url") or os.environ.get("OPENAI_BASE_URL")

        ai_config = AIConfig(base_url=base_url, api_key=api_key) if base_url else None
        agent = AIAgent(config=ai_config, tools=self._tools, name=self._name)

        # Build chat from history
        chat = Chat()
        for msg in session.build_history():
            chat._messages.append(msg)
        chat._messages.append({"role": "user", "content": message.text})

        # Run in thread to avoid blocking
        def _run():
            events = []
            for event in agent.forward(chat, model=session.model):
                events.append(event)
            return events

        try:
            events = await asyncio.to_thread(_run)
        except Exception as e:
            yield ErrorEvent(error_type="execution_error", message=str(e))
            return

        for event in events:
            mapped = self._map_event(event, Text, ToolCall, ToolResult, StepResult)
            if mapped:
                yield mapped

    def _map_event(self, event: Any, Text, ToolCall, ToolResult, StepResult) -> AgentEvent | None:
        if isinstance(event, Text):
            return DeltaEvent(content=event.content)

        elif isinstance(event, ToolCall):
            return ToolStartEvent(
                tool_call_id=event.id,
                tool_name=event.name,
                display_name=event.name,
                input_args=event.arguments,
            )

        elif isinstance(event, ToolResult):
            return ToolCompleteEvent(
                tool_call_id=event.id,
                tool_name=event.name,
                success=True,
                result=event.result[:1000] if event.result else "",
            )

        elif isinstance(event, StepResult):
            return UsageEvent(
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
            )

        return None

    async def stop(self, session: AgentSession) -> None:
        pass  # ai.py doesn't support mid-run cancellation yet
