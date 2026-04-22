"""Multi-agent orchestrator provider — supervisor pattern with parallel workers."""
from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType


class OrchestratorProvider(BaseProvider):
    name = "claudeagents"
    display_name = "Multi-Agent Orchestrator"
    supports_streaming = True
    supports_tools = True
    supports_sessions = False
    supports_agents = True

    def __init__(self):
        self._stop = False

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        """Supervisor loop: plan → parallel workers → review → repeat."""
        self._stop = False

        # Import Claude provider for worker execution
        from .claude import ClaudeProvider
        from .registry import get_provider

        cwd = config.cwd or "."
        mode = config.mode or "bypassPermissions"

        # Step 1: Plan subtasks using supervisor
        yield ProviderEvent(
            type=EventType.TOOL_START,
            metadata={"title": "🎯 Supervisor", "args": "Planning subtasks…"},
        )

        plan_prompt = (
            "You are a task supervisor. Break this into independent subtasks for parallel execution.\n"
            f'Respond ONLY with a JSON array: [{{"id":"1","prompt":"...","model":"haiku"}}]\n'
            f"Use 'haiku' for most tasks. Use 'sonnet' only for complex reasoning.\n\n"
            f"User request: {prompt}\nWorking directory: {cwd}"
        )

        try:
            supervisor = ClaudeProvider()
            plan_config = ProviderConfig(mode=mode, cwd=cwd)
            plan_text = ""
            async for event in supervisor.run(plan_prompt, plan_config):
                if event.type == EventType.TEXT:
                    plan_text += event.content
                elif event.type == EventType.COST:
                    yield event

            # Parse plan
            plan_text = plan_text.strip()
            if plan_text.startswith("```"):
                plan_text = "\n".join(plan_text.split("\n")[1:])
                if plan_text.endswith("```"):
                    plan_text = plan_text[:-3]

            try:
                parsed = json.loads(plan_text.strip())
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', plan_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                else:
                    parsed = [{"id": "1", "prompt": prompt, "model": "haiku"}]

            if isinstance(parsed, dict):
                subtasks = parsed.get("tasks", [parsed])
            else:
                subtasks = parsed

            yield ProviderEvent(
                type=EventType.TOOL_RESULT,
                content=f"Planned {len(subtasks)} subtask(s)",
                metadata={"title": "🎯 Plan", "is_error": False},
            )

            # Step 2: Run workers in parallel
            MODEL_MAP = {
                "haiku": "claude-haiku-4-5-20251001",
                "sonnet": "claude-sonnet-4-6",
            }

            for st in subtasks:
                worker_id = st.get("id", "1")
                worker_prompt = st.get("prompt", prompt)
                model_key = st.get("model", "haiku")
                model = MODEL_MAP.get(model_key, MODEL_MAP["haiku"])

                yield ProviderEvent(
                    type=EventType.AGENT_GROUP,
                    metadata={
                        "agent_id": worker_id,
                        "agent_label": f"Agent {worker_id} ({model_key})",
                        "model": model,
                        "status": "running",
                        "children": [],
                    },
                )

            # Execute workers
            async def run_worker(st):
                worker_id = st.get("id", "1")
                worker_prompt = st.get("prompt", prompt)
                model_key = st.get("model", "haiku")
                model = MODEL_MAP.get(model_key, MODEL_MAP["haiku"])

                worker = ClaudeProvider()
                worker_config = ProviderConfig(
                    mode=mode, cwd=cwd,
                    extra={"model_override": model},
                )

                result_text = ""
                events = []
                async for event in worker.run(worker_prompt, worker_config):
                    if event.type == EventType.TEXT:
                        result_text += event.content
                    events.append(event)

                return {"id": worker_id, "text": result_text, "events": events, "model": model_key}

            results = await asyncio.gather(*[run_worker(st) for st in subtasks])

            # Update agent groups with results
            for r in results:
                yield ProviderEvent(
                    type=EventType.AGENT_GROUP,
                    metadata={
                        "agent_id": r["id"],
                        "agent_label": f"Agent {r['id']} ({r['model']})",
                        "model": r["model"],
                        "status": "done",
                        "children": [{"role": "assistant", "content": r["text"]}],
                    },
                )

            # Step 3: Summarize
            summary_parts = []
            for r in results:
                summary_parts.append(f"Agent {r['id']}: {r['text'][:200]}")

            yield ProviderEvent(
                type=EventType.TEXT,
                content=f"All {len(results)} agent(s) completed.\n\n" + "\n\n".join(summary_parts),
            )

        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True
