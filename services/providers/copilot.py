"""GitHub Copilot provider — uses gh copilot CLI or Copilot API."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class CopilotProvider(BaseProvider):
    name = "copilot"
    display_name = "GitHub Copilot"
    supports_streaming = True
    supports_tools = False
    supports_sessions = False

    def __init__(self):
        self._stop = False
        self._proc = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False

        # Try copilot-cli first, fall back to gh copilot suggest
        cmd = self._build_command(prompt, config)

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if cmd[0] == "gh":
                # gh copilot suggest reads from stdin
                self._proc.stdin.write(prompt.encode())
                await self._proc.stdin.drain()
                self._proc.stdin.close()

            stdout, stderr = await self._proc.communicate()
            output = stdout.decode().strip()

            if self._stop:
                yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                yield ProviderEvent(type=EventType.DONE)
                return

            if self._proc.returncode != 0:
                err = stderr.decode().strip() or "Copilot command failed"
                yield ProviderEvent(type=EventType.ERROR, content=err)
                return

            if output:
                yield ProviderEvent(type=EventType.TEXT, content=output)

        except FileNotFoundError:
            yield ProviderEvent(
                type=EventType.ERROR,
                content="GitHub Copilot CLI not found. Install: gh extension install github/gh-copilot",
            )
            return
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    def _build_command(self, prompt: str, config: ProviderConfig) -> list[str]:
        """Build the copilot CLI command."""
        # Use GitHub Copilot in CLI via gh extension
        return ["gh", "copilot", "suggest", "-t", "shell", prompt]

    async def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()

    async def health_check(self):
        import shutil
        has_gh = shutil.which("gh") is not None
        if not has_gh:
            return {"status": "unavailable", "provider": self.name, "reason": "gh CLI not installed"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh", "copilot", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return {"status": "ok" if proc.returncode == 0 else "unavailable", "provider": self.name}
        except Exception:
            return {"status": "unavailable", "provider": self.name}
