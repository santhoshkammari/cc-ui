"""
ChatAgent — lightweight direct LLM chat (OpenAI-compatible).

No tools, no agents, just clean conversation.
Good for brainstorming, Q&A, code review discussions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from lib.agents.base import BaseAgent
from lib.agents.events import (
    AgentEvent,
    DeltaEvent,
    ErrorEvent,
    UsageEvent,
)
from lib.agents.session import AgentSession, UserMessage

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """
    Simple chat agent using OpenAI-compatible API.
    Streams text responses. No tool use.
    """

    def __init__(
        self,
        agent_id: str = "chat",
        agent_name: str = "Chat",
        default_model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._id = agent_id
        self._name = agent_name
        self._default_model = default_model
        self._base_url = base_url
        self._api_key = api_key

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Direct LLM chat — brainstorm, discuss, review."

    @property
    def icon(self) -> str:
        return "💬"

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def supports_tools(self) -> bool:
        return False

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

        try:
            from openai import AsyncOpenAI
        except ImportError:
            yield ErrorEvent(error_type="import_error", message="openai package not installed")
            return

        api_key = self._api_key or session.config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = self._base_url or session.config.get("base_url") or os.environ.get("OPENAI_BASE_URL")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)

        messages = session.build_history()
        messages.append({"role": "user", "content": message.text})

        try:
            stream = await client.chat.completions.create(
                model=session.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield DeltaEvent(content=delta.content)

                if chunk.usage:
                    yield UsageEvent(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

        except Exception as e:
            yield ErrorEvent(error_type="api_error", message=str(e))

    async def stop(self, session: AgentSession) -> None:
        pass  # OpenAI streaming doesn't support cancel easily
