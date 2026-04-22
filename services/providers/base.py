"""
Base provider protocol for all AI integrations.

Every provider implements this interface so the system can swap
between Claude, Gemini, Copilot, Qwen, VLLM, etc. with zero friction.
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional


class EventType(str, Enum):
    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    AGENT_GROUP = "agent_group"
    ERROR = "error"
    DONE = "done"
    COST = "cost"


@dataclass
class ProviderEvent:
    """Unified event emitted by all providers."""
    type: EventType
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_history_entry(self) -> dict | None:
        """Convert to a history message dict for storage."""
        if self.type == EventType.TEXT:
            return {"role": "assistant", "content": self.content}
        elif self.type == EventType.TOOL_START:
            title = self.metadata.get("title", "⚙ tool")
            args = self.metadata.get("args", "")
            return {
                "role": "assistant",
                "content": f"```json\n{args}\n```" if args else "",
                "metadata": {"title": title, "status": "pending"},
            }
        elif self.type == EventType.TOOL_RESULT:
            title = self.metadata.get("title", "✓ result")
            is_error = self.metadata.get("is_error", False)
            icon = "❌" if is_error else "✓"
            preview = self.content[:600] + ("…" if len(self.content) > 600 else "")
            return {
                "role": "assistant",
                "content": f"```\n{preview}\n```",
                "metadata": {"title": f"{icon} result", "status": "done"},
            }
        elif self.type == EventType.THINKING:
            return {
                "role": "assistant",
                "content": self.content[:500],
                "metadata": {"title": "💭 Thinking", "status": "done"},
            }
        elif self.type == EventType.ERROR:
            return {"role": "assistant", "content": f"❌ {self.content}"}
        elif self.type == EventType.AGENT_GROUP:
            return {
                "role": "agent-group",
                "agent_id": self.metadata.get("agent_id", ""),
                "agent_label": self.metadata.get("agent_label", "Agent"),
                "model": self.metadata.get("model", ""),
                "status": self.metadata.get("status", "running"),
                "children": self.metadata.get("children", []),
            }
        return None


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""
    model: str = ""
    mode: str = "default"
    cwd: str = ""
    session_id: str | None = None
    api_key: str = ""
    base_url: str = ""
    extra: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base for all AI provider adapters."""

    name: str = "base"
    display_name: str = "Base Provider"
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_sessions: bool = False
    supports_agents: bool = False

    @abstractmethod
    async def run(
        self,
        prompt: str,
        config: ProviderConfig,
        history: list[dict] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Execute a prompt and yield events."""
        ...

    async def stop(self) -> None:
        """Signal the provider to stop current execution."""
        pass

    async def health_check(self) -> dict:
        """Check if the provider is available and responsive."""
        return {"status": "unknown", "provider": self.name}

    def get_capabilities(self) -> dict:
        """Return provider capabilities for the UI."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "streaming": self.supports_streaming,
            "tools": self.supports_tools,
            "sessions": self.supports_sessions,
            "agents": self.supports_agents,
        }
