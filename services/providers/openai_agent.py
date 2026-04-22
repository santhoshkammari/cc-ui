"""OpenAI Agent SDK provider — uses openai-agents package."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class OpenAIAgentProvider(BaseProvider):
    name = "openai-agent"
    display_name = "OpenAI Agent"
    supports_streaming = True
    supports_tools = True
    supports_sessions = False
    supports_agents = True

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        model = config.model or "gpt-4o"

        if not api_key:
            yield ProviderEvent(type=EventType.ERROR, content="OPENAI_API_KEY not set")
            return

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)

            messages = []
            if history:
                for h in history:
                    if h.get("role") in ("user", "assistant", "system") and not h.get("metadata"):
                        messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": prompt})

            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=4096,
            )

            text_buf = ""
            async for chunk in stream:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text_buf += delta.content
                    yield ProviderEvent(type=EventType.TEXT, content=delta.content)

                if chunk.choices and chunk.choices[0].finish_reason:
                    break

        except ImportError:
            yield ProviderEvent(type=EventType.ERROR, content="openai package required: pip install openai")
            return
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return {"status": "ok" if api_key else "no_api_key", "provider": self.name}
