"""GitHub Copilot provider — uses the official copilot-sdk Python package.

Supports streaming, tool use, sessions, and multiple models
(gpt-5.4, gpt-5-mini, claude-sonnet-4.6, claude-haiku-4.5, claude-opus-4.6).

Requires: pip install github-copilot-sdk
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType

log = logging.getLogger("cc-ui.copilot")

# Default model when none specified
DEFAULT_MODEL = "gpt-5.4"


class CopilotProvider(BaseProvider):
    name = "copilot"
    display_name = "GitHub Copilot"
    description = "GitHub Copilot SDK with full agentic capabilities. Streaming, tool use, sessions, and multi-model support (GPT-5.4, GPT-5 mini, Claude Sonnet/Haiku/Opus). Runs the bundled Copilot CLI under the hood."
    supports_streaming = True
    supports_tools = True
    supports_sessions = True
    supports_agents = True

    def __init__(self):
        self._stop = False
        self._client = None
        self._session = None

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        # Lazy import — only fail when actually used
        try:
            # Ensure the SDK path is importable
            sdk_path = None
            try:
                import copilot as _cp
            except ImportError:
                import glob
                paths = glob.glob("/home/ntlpt24/.local/lib/python*/site-packages")
                for p in paths:
                    if p not in sys.path:
                        sys.path.insert(0, p)
                        sdk_path = p

            from copilot import CopilotClient
            from copilot.session import PermissionHandler
            from copilot.generated.session_events import SessionEventType
        except ImportError as e:
            yield ProviderEvent(
                type=EventType.ERROR,
                content=f"Copilot SDK not installed: {e}\nInstall with: pip install github-copilot-sdk",
            )
            return

        self._stop = False
        model = config.model or config.extra.get("copilot_model") or DEFAULT_MODEL
        session_id = config.session_id
        text_buf = ""
        total_cost = 0.0
        usage = {}
        resolved_session_id = None

        try:
            self._client = CopilotClient()
            await self._client.start()

            # Create or resume session
            session_kwargs = dict(
                on_permission_request=PermissionHandler.approve_all,
                model=model,
                streaming=True,
            )

            if session_id:
                self._session = await self._client.resume_session(
                    session_id,
                    on_permission_request=PermissionHandler.approve_all,
                )
            else:
                self._session = await self._client.create_session(**session_kwargs)

            resolved_session_id = getattr(self._session, 'session_id', None) or session_id

            # Collect events via callback
            done_event = asyncio.Event()
            events_queue: asyncio.Queue = asyncio.Queue()

            def on_event(event):
                events_queue.put_nowait(event)
                etype = getattr(event, 'type', None)
                if etype and etype.value in ('session.idle', 'session.error', 'session.shutdown'):
                    done_event.set()

            self._session.on(on_event)
            await self._session.send(prompt)

            # Process events until done
            while not done_event.is_set() or not events_queue.empty():
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    break

                try:
                    event = await asyncio.wait_for(events_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                etype = getattr(event, 'type', None)
                if not etype:
                    continue

                ev_type = etype.value
                data = getattr(event, 'data', None)

                if ev_type == 'assistant.message_delta' and data:
                    delta = getattr(data, 'delta_content', '') or ''
                    if delta:
                        text_buf += delta
                        yield ProviderEvent(type=EventType.TEXT, content=delta)

                elif ev_type == 'assistant.reasoning_delta' and data:
                    delta = getattr(data, 'delta_content', '') or ''
                    if delta:
                        yield ProviderEvent(type=EventType.THINKING, content=delta)

                elif ev_type == 'tool.execution_start' and data:
                    tool_name = getattr(data, 'tool_name', '') or 'tool'
                    args = getattr(data, 'input', '') or ''
                    yield ProviderEvent(
                        type=EventType.TOOL_START,
                        metadata={"title": f"⚙ {tool_name}", "args": str(args)[:500]},
                    )

                elif ev_type == 'tool.execution_complete' and data:
                    output = getattr(data, 'output', '') or ''
                    is_error = getattr(data, 'is_error', False)
                    yield ProviderEvent(
                        type=EventType.TOOL_RESULT,
                        content=str(output)[:600],
                        metadata={"is_error": bool(is_error)},
                    )

                elif ev_type == 'assistant.usage' and data:
                    u = getattr(data, 'usage', None)
                    if u:
                        usage["input_tokens"] = getattr(u, 'input_tokens', 0) or 0
                        usage["output_tokens"] = getattr(u, 'output_tokens', 0) or 0

                elif ev_type == 'subagent.started' and data:
                    agent_id = getattr(data, 'session_id', '') or ''
                    label = getattr(data, 'label', '') or f'Sub-agent {agent_id[:8]}'
                    yield ProviderEvent(
                        type=EventType.AGENT_GROUP,
                        metadata={"agent_id": agent_id, "agent_label": label, "status": "running", "children": []},
                    )

                elif ev_type == 'subagent.completed' and data:
                    agent_id = getattr(data, 'session_id', '') or ''
                    yield ProviderEvent(
                        type=EventType.AGENT_GROUP,
                        metadata={"agent_id": agent_id, "status": "done"},
                    )

                elif ev_type == 'session.error' and data:
                    msg = getattr(data, 'message', '') or str(data)
                    yield ProviderEvent(type=EventType.ERROR, content=f"Copilot error: {msg}")
                    break

                elif ev_type == 'session.idle':
                    break

        except Exception as e:
            log.exception("Copilot provider error: %s", e)
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
        finally:
            # Cleanup
            try:
                if self._session:
                    await self._session.disconnect()
                if self._client:
                    await self._client.stop()
            except Exception:
                pass
            self._session = None
            self._client = None

        # Emit cost/session info
        if resolved_session_id or usage:
            yield ProviderEvent(
                type=EventType.COST,
                metadata={
                    "session_id": resolved_session_id,
                    "total_cost_usd": total_cost,
                    "usage": usage,
                },
            )

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True
        if self._session:
            try:
                await self._session.disconnect()
            except Exception:
                pass

    async def health_check(self):
        try:
            sdk_ok = False
            try:
                import copilot
                sdk_ok = True
            except ImportError:
                import sys, glob
                for p in glob.glob("/home/ntlpt24/.local/lib/python*/site-packages"):
                    if p not in sys.path:
                        sys.path.insert(0, p)
                from copilot import CopilotClient
                sdk_ok = True

            if sdk_ok:
                return {"status": "ok", "provider": self.name, "sdk": "github-copilot-sdk"}
        except Exception as e:
            return {"status": "unavailable", "provider": self.name, "error": str(e)}
        return {"status": "unavailable", "provider": self.name}
