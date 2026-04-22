"""Claude Code provider — uses claude_agent_sdk for streaming."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class ClaudeProvider(BaseProvider):
    name = "claude"
    display_name = "Claude Code"
    description = "Anthropic's agentic coding assistant. Reads/writes files, runs shell commands, manages git, and reasons through complex multi-step tasks. Supports session resumption and sub-agent orchestration."
    supports_streaming = True
    supports_tools = True
    supports_sessions = True
    supports_agents = True

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions,
            AssistantMessage, ResultMessage, SystemMessage,
            TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,
        )

        self._stop = False
        opts = ClaudeAgentOptions(
            permission_mode=config.mode or "bypassPermissions",
            resume=config.session_id,
            cwd=config.cwd or None,
            env={**os.environ, **config.extra.get("env", {})},
            model=config.extra.get("model_override"),
        )

        text_buf = ""
        try:
            async for msg in query(prompt=prompt, options=opts):
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                if isinstance(msg, AssistantMessage):
                    if msg.error:
                        yield ProviderEvent(type=EventType.ERROR, content=str(msg.error))
                        return
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_buf += block.text
                            yield ProviderEvent(type=EventType.TEXT, content=block.text)
                        elif isinstance(block, ThinkingBlock):
                            yield ProviderEvent(type=EventType.THINKING, content=block.thinking[:500])
                        elif isinstance(block, ToolUseBlock):
                            args = json.dumps(block.input, indent=2) if block.input else ""
                            yield ProviderEvent(
                                type=EventType.TOOL_START,
                                metadata={"title": f"⚙ {block.name}", "args": args},
                            )
                        elif isinstance(block, ToolResultBlock):
                            content = block.content or ""
                            if isinstance(content, list):
                                content = "\n".join(c.get("text", str(c)) for c in content)
                            yield ProviderEvent(
                                type=EventType.TOOL_RESULT,
                                content=str(content),
                                metadata={"is_error": block.is_error},
                            )

                elif isinstance(msg, ResultMessage):
                    yield ProviderEvent(
                        type=EventType.COST,
                        metadata={
                            "session_id": msg.session_id,
                            "total_cost_usd": msg.total_cost_usd or 0,
                            "usage": msg.usage or {},
                        },
                    )

        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        import shutil
        has_claude = shutil.which("claude") is not None
        return {
            "status": "ok" if has_claude else "unavailable",
            "provider": self.name,
            "binary": "claude" if has_claude else None,
        }
