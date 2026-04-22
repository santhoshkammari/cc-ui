"""OpenCode provider — wraps `opencode run --format json`."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class OpenCodeProvider(BaseProvider):
    name = "opencode"
    display_name = "OpenCode"
    supports_streaming = True
    supports_tools = True
    supports_sessions = True

    def __init__(self):
        self._stop = False
        self._proc = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False
        cmd = ["opencode", "run", "--format", "json"]
        if config.session_id:
            cmd += ["--session", config.session_id]
        if config.cwd:
            cmd += ["--dir", config.cwd]
        cmd += [prompt]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async for line in self._proc.stdout:
                if self._stop:
                    self._proc.terminate()
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                line = line.decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue

                session_id = msg.get("sessionID")
                mtype = msg.get("type")
                part = msg.get("part", {})

                if mtype == "text":
                    yield ProviderEvent(type=EventType.TEXT, content=part.get("text", ""))

                elif mtype == "tool_use":
                    state = part.get("state", {})
                    tool_name = part.get("tool", "tool")
                    title = state.get("title") or tool_name
                    args = json.dumps(state.get("input", {}), indent=2)
                    output = state.get("output", "")
                    if isinstance(output, list):
                        output = "\n".join(c.get("text", str(c)) for c in output)
                    is_error = state.get("metadata", {}).get("exit", 0) != 0

                    yield ProviderEvent(
                        type=EventType.TOOL_START,
                        metadata={"title": f"⚙ {title}", "args": args},
                    )
                    yield ProviderEvent(
                        type=EventType.TOOL_RESULT,
                        content=str(output),
                        metadata={"is_error": is_error},
                    )

                if session_id:
                    yield ProviderEvent(
                        type=EventType.COST,
                        metadata={"session_id": session_id},
                    )

            await self._proc.wait()
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()

    async def health_check(self):
        import shutil
        has_bin = shutil.which("opencode") is not None
        return {"status": "ok" if has_bin else "unavailable", "provider": self.name}
