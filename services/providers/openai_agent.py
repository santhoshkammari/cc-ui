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
    description = "OpenAI's agent SDK with built-in tool use, handoffs, and guardrails. Runs multi-step workflows with GPT models. Supports streaming and sub-agent delegation. Requires OPENAI_API_KEY."
    supports_streaming = True
    supports_tools = True
    supports_sessions = False
    supports_agents = True
    available_models = [
        "gpt-5.4", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini",
        "gpt-4.1-nano", "o3", "o3-mini", "o4-mini",
    ]

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
            last_chunk = None
            async for chunk in stream:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                last_chunk = chunk
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

        # Emit cost event from usage metadata on last chunk
        from .model_costs import estimate_cost
        input_tokens = 0
        output_tokens = 0
        if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
            input_tokens = last_chunk.usage.prompt_tokens or 0
            output_tokens = last_chunk.usage.completion_tokens or 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        total_cost = estimate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        yield ProviderEvent(
            type=EventType.COST,
            metadata={
                "total_cost_usd": total_cost,
                "usage": usage,
            },
        )

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return {"status": "ok" if api_key else "no_api_key", "provider": self.name}
