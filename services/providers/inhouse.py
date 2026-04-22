"""In-house AI framework provider — wraps lib/ai.py AIAgent."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class InhouseProvider(BaseProvider):
    name = "inhouse"
    display_name = "AI Framework (In-house)"
    supports_streaming = True
    supports_tools = True
    supports_sessions = False

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False

        base_url = config.base_url or config.extra.get("base_url") or os.environ.get("OPENAI_BASE_URL", "http://localhost:9999/v1")
        api_key = config.api_key or config.extra.get("api_key") or "dummy"
        model = config.model or config.extra.get("model") or ""
        mode = config.extra.get("ai_mode") or "instruct_general"
        use_tools = config.extra.get("use_tools", False)

        try:
            # Import from our lib
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lib_path = os.path.join(here, "lib")
            if lib_path not in sys.path:
                sys.path.insert(0, lib_path)

            from ai import AIAgent, AIConfig, Chat, Text, ToolCall, ToolResult, StepResult, AgentResult, DoneEvent
            from ai import read_file, write_file, edit_file, bash_run, glob_files, grep_files

            ai_config = AIConfig(base_url=base_url, api_key=api_key)
            tools = None
            if use_tools:
                tools = [read_file, write_file, edit_file, bash_run, glob_files, grep_files]
            agent = AIAgent(config=ai_config, tools=tools)

            # Build chat with history
            messages = []
            if history:
                for h in history:
                    if h.get("role") in ("user", "assistant") and not h.get("metadata"):
                        messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": prompt})

            chat = Chat.__new__(Chat)
            chat._messages = messages

            # Run in thread to avoid blocking
            def _run_sync():
                events = []
                if use_tools:
                    for event in agent.forward(chat, model=model, mode=mode):
                        events.append(event)
                else:
                    for event in agent.step(chat, model=model, mode=mode):
                        events.append(event)
                return events

            events = await asyncio.to_thread(_run_sync)

            for event in events:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                if isinstance(event, Text):
                    yield ProviderEvent(type=EventType.TEXT, content=event.content)
                elif isinstance(event, ToolCall):
                    yield ProviderEvent(
                        type=EventType.TOOL_START,
                        metadata={"title": f"⚙ {event.name}", "args": event.arguments},
                    )
                elif isinstance(event, ToolResult):
                    yield ProviderEvent(
                        type=EventType.TOOL_RESULT,
                        content=event.result[:600],
                        metadata={"is_error": False},
                    )
                elif isinstance(event, StepResult):
                    yield ProviderEvent(
                        type=EventType.COST,
                        metadata={
                            "usage": {
                                "input_tokens": event.input_tokens or 0,
                                "output_tokens": event.output_tokens or 0,
                            },
                        },
                    )

        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        try:
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lib_path = os.path.join(here, "lib")
            if os.path.exists(os.path.join(lib_path, "ai.py")):
                return {"status": "ok", "provider": self.name}
            return {"status": "missing_lib", "provider": self.name}
        except Exception:
            return {"status": "error", "provider": self.name}
