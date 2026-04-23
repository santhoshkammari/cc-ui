"""vLLM provider — routes through Claude SDK with env overrides, or direct OpenAI API."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType

DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_VLLM_MODEL = "/home/ng6309/datascience/santhosh/models/qwen3.5-9b"


class VLLMProvider(BaseProvider):
    name = "vllm"
    display_name = "vLLM (Local)"
    description = "Self-hosted models via vLLM's OpenAI-compatible API. Run any HuggingFace model locally with GPU acceleration. Configure IP, port, and model in Settings."
    supports_streaming = True
    supports_tools = False
    supports_sessions = False
    available_models = ["auto"]

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        """Direct OpenAI-compatible API call to vLLM server."""
        self._stop = False
        base_url = config.base_url or config.extra.get("vllm_url") or DEFAULT_VLLM_URL
        api_key = config.api_key or config.extra.get("vllm_key") or "dummy"
        model = config.model or config.extra.get("vllm_model") or DEFAULT_VLLM_MODEL

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

            if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                yield ProviderEvent(
                    type=EventType.COST,
                    metadata={"usage": {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }},
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
        base_url = DEFAULT_VLLM_URL
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{base_url}/health")
                return {"status": "ok" if r.status_code == 200 else "error", "provider": self.name}
        except Exception:
            return {"status": "unavailable", "provider": self.name}
