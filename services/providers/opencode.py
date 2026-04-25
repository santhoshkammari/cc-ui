"""OpenCode provider — wraps `opencode run --format json`.

Event types from opencode CLI:
  step_start  — new LLM turn begins
  text        — streaming text chunk
  tool_use    — tool announced + completed (state has input/output/error)
  step_finish — step done with token/cost info
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class OpenCodeProvider(BaseProvider):
    name = "opencode"
    display_name = "OpenCode"
    description = "Open-source agentic coding CLI. File editing, shell access, and step-by-step reasoning with tool use. Supports persistent sessions and multiple LLM backends."
    supports_streaming = True
    supports_tools = True
    supports_sessions = True
    available_models = [
        "opencode/big-pickle", "opencode/gpt-5-nano", "opencode/ling-2.6-flash-free",
        "opencode/minimax-m2.5-free", "opencode/nemotron-3-super-free",
        "anthropic/claude-sonnet-4-20250514", "anthropic/claude-opus-4-20250514",
        "anthropic/claude-haiku-4.5-20241022",
        "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/o3", "openai/o4-mini",
        "google/gemini-2.5-flash", "google/gemini-2.5-pro",
    ]

    def __init__(self):
        self._stop = False
        self._proc = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False
        cmd = ["opencode", "run", "--format", "json"]
        if config.model:
            cmd += ["--model", config.model]
        if config.session_id:
            cmd += ["--session", config.session_id]
        if config.cwd:
            cmd += ["--dir", config.cwd]
        if config.extra.get("thinking"):
            cmd += ["--thinking"]
        if config.extra.get("fork"):
            cmd += ["--fork"]

        # File attachments via -f flag
        for f in config.extra.get("files", []):
            cmd += ["-f", f]

        cmd += [prompt]

        session_id = None
        total_cost = 0.0
        usage = {}

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )

            async for line in self._proc.stdout:
                if self._stop:
                    self._proc.terminate()
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                line = line.decode().strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue

                session_id = msg.get("sessionID", session_id)
                mtype = msg.get("type")
                part = msg.get("part", {})

                if mtype == "text":
                    yield ProviderEvent(type=EventType.TEXT, content=part.get("text", ""))

                elif mtype == "error":
                    error_data = msg.get("error", {})
                    error_msg = error_data.get("data", {}).get("message", "") or error_data.get("name", "Unknown error")
                    yield ProviderEvent(type=EventType.ERROR, content=f"OpenCode error: {error_msg}")
                    return

                elif mtype == "tool_use":
                    state = part.get("state", {})
                    tool_name = part.get("tool", "tool")
                    title = part.get("title", tool_name) or tool_name
                    args = json.dumps(state.get("input", {}), indent=2)
                    output = state.get("output", "") or state.get("error", "")
                    if isinstance(output, list):
                        output = "\n".join(c.get("text", str(c)) for c in output)
                    status = state.get("status", "")
                    is_error = status == "error" or bool(state.get("error"))

                    yield ProviderEvent(
                        type=EventType.TOOL_START,
                        metadata={"title": f"⚙ {title}", "args": args},
                    )
                    yield ProviderEvent(
                        type=EventType.TOOL_RESULT,
                        content=str(output),
                        metadata={"is_error": is_error},
                    )

                elif mtype == "step_finish":
                    tokens = part.get("tokens", {})
                    cache = tokens.get("cache", {})
                    step_cost = part.get("cost", 0.0)
                    total_cost += step_cost
                    usage["input_tokens"] = usage.get("input_tokens", 0) + tokens.get("input", 0)
                    usage["output_tokens"] = usage.get("output_tokens", 0) + tokens.get("output", 0)
                    usage["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0) + cache.get("read", 0)
                    usage["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens", 0) + cache.get("write", 0)

            await self._proc.wait()
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        if session_id or total_cost:
            yield ProviderEvent(
                type=EventType.COST,
                metadata={
                    "session_id": session_id,
                    "total_cost_usd": total_cost,
                    "usage": usage,
                },
            )

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()

    async def health_check(self):
        import shutil
        has_bin = shutil.which("opencode") is not None
        return {"status": "ok" if has_bin else "unavailable", "provider": self.name}
