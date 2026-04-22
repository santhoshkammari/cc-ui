"""Qwen Code provider — wraps `qwen` CLI with stream-json output."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType

SKIP_SUBTYPES = {"init", "system"}
APPROVAL_MAP = {
    "bypassPermissions": "yolo",
    "acceptEdits": "auto-edit",
    "dontAsk": "yolo",
    "default": "default",
    "plan": "plan",
}


class QwenProvider(BaseProvider):
    name = "qwen"
    display_name = "Qwen Code"
    supports_streaming = True
    supports_tools = True
    supports_sessions = True

    def __init__(self):
        self._stop = False
        self._proc = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False
        approval = APPROVAL_MAP.get(config.mode, "default")
        cmd = ["qwen", "--output-format", "stream-json", "--approval-mode", approval]
        if config.session_id:
            cmd += ["--resume", config.session_id]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=config.cwd or None,
            )
            self._proc.stdin.write(prompt.encode())
            await self._proc.stdin.drain()
            self._proc.stdin.close()

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

                mtype = msg.get("type")
                if mtype == "assistant":
                    for block in msg.get("message", {}).get("content", []):
                        btype = block.get("type")
                        if btype == "text":
                            yield ProviderEvent(type=EventType.TEXT, content=block.get("text", ""))
                        elif btype == "thinking":
                            yield ProviderEvent(type=EventType.THINKING, content=block.get("thinking", "")[:500])
                        elif btype == "tool_use":
                            args = json.dumps(block.get("input", {}), indent=2)
                            yield ProviderEvent(
                                type=EventType.TOOL_START,
                                metadata={"title": f"⚙ {block.get('name', '')}", "args": args},
                            )
                        elif btype == "tool_result":
                            content = block.get("content", "")
                            if isinstance(content, list):
                                content = "\n".join(c.get("text", str(c)) for c in content)
                            yield ProviderEvent(
                                type=EventType.TOOL_RESULT,
                                content=str(content),
                                metadata={"is_error": block.get("is_error", False)},
                            )

                elif mtype == "result":
                    yield ProviderEvent(
                        type=EventType.COST,
                        metadata={"session_id": msg.get("session_id")},
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
        return {"status": "ok" if shutil.which("qwen") else "unavailable", "provider": self.name}
