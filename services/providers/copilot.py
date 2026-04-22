"""GitHub Copilot provider — uses `gh copilot` CLI extension.

Copilot CLI modes:
  gh copilot suggest -t shell "prompt"   — shell command suggestion
  gh copilot suggest -t gh "prompt"      — gh CLI suggestion
  gh copilot explain "prompt"            — explain code/command

This is single-shot (non-streaming). For agentic use, consider
GitHub Copilot Coding Agent via the API instead.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class CopilotProvider(BaseProvider):
    name = "copilot"
    display_name = "GitHub Copilot"
    supports_streaming = False
    supports_tools = False
    supports_sessions = False

    def __init__(self):
        self._stop = False
        self._proc = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        self._stop = False

        import shutil
        if not shutil.which("gh"):
            yield ProviderEvent(type=EventType.ERROR, content="gh CLI not installed")
            return

        # Check if copilot extension is available
        check = await asyncio.create_subprocess_exec(
            "gh", "copilot", "--help",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, check_err = await check.communicate()
        if check.returncode != 0:
            yield ProviderEvent(
                type=EventType.ERROR,
                content="GitHub Copilot extension not installed.\n"
                        "Install with: `gh extension install github/gh-copilot`\n"
                        "Then authenticate: `gh auth refresh -s copilot`",
            )
            return

        # Determine mode from extra config
        copilot_mode = config.extra.get("copilot_mode", "suggest")
        target_type = config.extra.get("copilot_type", "shell")

        if copilot_mode == "explain":
            cmd = ["gh", "copilot", "explain", prompt]
        else:
            cmd = ["gh", "copilot", "suggest", "-t", target_type, prompt]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env={**os.environ, "GH_FORCE_TTY": "0"},
            )

            stdout, stderr = await self._proc.communicate()

            if self._stop:
                yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                yield ProviderEvent(type=EventType.DONE)
                return

            output = stdout.decode().strip()
            err_output = stderr.decode().strip()

            if self._proc.returncode != 0:
                yield ProviderEvent(
                    type=EventType.ERROR,
                    content=err_output or f"Copilot exited with code {self._proc.returncode}",
                )
                return

            if output:
                yield ProviderEvent(type=EventType.TEXT, content=output)
            elif err_output:
                yield ProviderEvent(type=EventType.TEXT, content=err_output)
            else:
                yield ProviderEvent(type=EventType.TEXT, content="*(no suggestion)*")

        except FileNotFoundError:
            yield ProviderEvent(
                type=EventType.ERROR,
                content="gh CLI not found in PATH",
            )
            return
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
        has_gh = shutil.which("gh") is not None
        if not has_gh:
            return {"status": "unavailable", "provider": self.name, "reason": "gh CLI not installed"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh", "copilot", "--help",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return {
                "status": "ok" if proc.returncode == 0 else "unavailable",
                "provider": self.name,
                "note": "extension missing" if proc.returncode != 0 else None,
            }
        except Exception:
            return {"status": "unavailable", "provider": self.name}
