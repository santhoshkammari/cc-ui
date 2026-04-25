"""Kivi provider — local model via OpenAI-compatible API (same as vLLM but separate config)."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType

DEFAULT_KIVI_URL = "http://localhost:9999"
DEFAULT_KIVI_MODEL = "mock-model-v1"


class KiviProvider(BaseProvider):
    name = "kivi"
    display_name = "Kivi (Local)"
    description = "Lightweight local inference via OpenAI-compatible API. Similar to vLLM but for smaller models. Supports streaming and tool-use formatted responses."
    supports_streaming = True
    supports_tools = True
    supports_sessions = False
    available_models = ["auto"]

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False
        base_url = config.base_url or config.extra.get("kivi_url") or DEFAULT_KIVI_URL
        api_key = config.api_key or config.extra.get("kivi_key") or "dummy"
        model = config.model or config.extra.get("kivi_model") or DEFAULT_KIVI_MODEL

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        messages = []
        if history:
            for h in history:
                if h.get("role") in ("user", "assistant") and not h.get("metadata"):
                    messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=api_key)

            tools = None
            if config.extra.get("tools"):
                tools = config.extra["tools"]

            kwargs = {
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": 4096,
            }
            if tools:
                kwargs["tools"] = tools

            stream = await client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta:
                    if delta.content:
                        yield ProviderEvent(type=EventType.TEXT, content=delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.function and tc.function.name:
                                yield ProviderEvent(
                                    type=EventType.TOOL_START,
                                    metadata={"title": f"⚙ {tc.function.name}", "args": tc.function.arguments or ""},
                                )

                if chunk.choices and chunk.choices[0].finish_reason:
                    break

            # Emit cost event (self-hosted = $0)
            yield ProviderEvent(
                type=EventType.COST,
                metadata={"total_cost_usd": 0.0, "usage": {}},
            )

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
        base_url = DEFAULT_KIVI_URL
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{base_url}/health")
                return {"status": "ok" if r.status_code == 200 else "error", "provider": self.name}
        except Exception:
            return {"status": "unavailable", "provider": self.name}
