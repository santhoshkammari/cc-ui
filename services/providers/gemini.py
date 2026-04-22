"""Google Gemini provider — uses google-genai SDK."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class GeminiProvider(BaseProvider):
    name = "gemini"
    display_name = "Google Gemini"
    description = "Google's multimodal AI. Fast streaming text generation via the Gemini API. Best for quick questions, summaries, and brainstorming. Requires GEMINI_API_KEY."
    supports_streaming = True
    supports_tools = False
    supports_sessions = False
    available_models = [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
        "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash",
    ]

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False

        api_key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        model = config.model or "gemini-2.5-flash"

        if not api_key:
            yield ProviderEvent(type=EventType.ERROR, content="GEMINI_API_KEY not set")
            return

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            contents = []
            if history:
                for h in history:
                    if h.get("role") in ("user", "assistant") and not h.get("metadata"):
                        role = "user" if h["role"] == "user" else "model"
                        contents.append(types.Content(role=role, parts=[types.Part(text=h["content"])]))
            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

            response = await asyncio.to_thread(
                lambda: client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )
            )

            text_buf = ""
            for chunk in response:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                if chunk.text:
                    text_buf += chunk.text
                    yield ProviderEvent(type=EventType.TEXT, content=chunk.text)

        except ImportError:
            yield ProviderEvent(
                type=EventType.ERROR,
                content="google-genai package required: pip install google-genai",
            )
            return
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        return {
            "status": "ok" if api_key else "no_api_key",
            "provider": self.name,
        }
