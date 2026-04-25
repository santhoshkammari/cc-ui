"""vLLM provider — OpenAI-compatible API with agentic tool-calling loop."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator

from .base import BaseProvider, ProviderConfig, ProviderEvent, EventType

log = logging.getLogger(__name__)

DEFAULT_VLLM_URL = "http://192.168.170.76:8000"
DEFAULT_VLLM_MODEL = ""  # empty = vLLM auto-picks the loaded model
MAX_TOOL_ITERATIONS = 1248  # safety limit for agentic loop


class VLLMProvider(BaseProvider):
    name = "vllm"
    display_name = "vLLM (Local)"
    description = "Self-hosted models via vLLM's OpenAI-compatible API. Run any HuggingFace model locally with GPU acceleration. Configure IP, port, and model in Settings."
    supports_streaming = True
    supports_tools = True
    supports_sessions = False
    available_models = ["auto"]

    def __init__(self):
        self._stop = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _discover_model(self, base_url: str) -> str | None:
        """Auto-discover model from server /v1/models endpoint."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as hc:
                r = await hc.get(f"{base_url}/models")
                data = r.json()
                if data.get("data"):
                    return data["data"][0]["id"]
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_thinking(content: str) -> tuple[str, str]:
        """Separate thinking from visible content for Qwen3 and similar models.
        
        Handles multiple formats:
        - <think>X</think>Y  → thinking=X, content=Y
        - X</think>Y         → thinking=X, content=Y (Qwen3 tool-call format)
        - <think>X (unclosed) → thinking=X, content=""
        
        Returns (thinking_text, clean_content).
        """
        if not content:
            return "", ""
        
        # Case 1: Full <think>...</think> pairs
        if '<think>' in content:
            thinking_parts = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # Handle unclosed <think> at end
            unclosed = re.search(r'<think>(.*?)$', clean, re.DOTALL)
            if unclosed:
                thinking_parts.append(unclosed.group(1))
                clean = re.sub(r'<think>.*?$', '', clean, flags=re.DOTALL)
            clean = clean.strip()
            return "\n".join(thinking_parts).strip(), clean
        
        # Case 2: Bare </think> (Qwen3 pattern — thinking before tag, content after)
        if '</think>' in content:
            parts = content.split('</think>', 1)
            thinking = parts[0].strip()
            clean = parts[1].strip() if len(parts) > 1 else ""
            return thinking, clean
        
        # Case 3: No thinking markers
        return "", content.strip()

    def _build_messages(self, prompt: str, history: list | None, cwd: str) -> list[dict]:
        """Build the messages list with system prompt, history, and user prompt."""
        from services.tools.tool_manager import get_system_prompt

        messages = [{"role": "system", "content": get_system_prompt(cwd)}]
        if history:
            for h in history:
                if h.get("role") in ("user", "assistant") and not h.get("metadata"):
                    messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _execute_tool(self, name: str, arguments: dict, cwd: str) -> tuple[str, bool]:
        """Run a tool in a thread pool to avoid blocking the event loop."""
        from services.tools.tool_manager import execute_tool
        return await asyncio.get_event_loop().run_in_executor(
            None, execute_tool, name, arguments, cwd
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self, prompt: str, config: ProviderConfig, history=None) -> AsyncIterator[ProviderEvent]:
        """Agentic loop: stream text, detect tool calls, execute, repeat."""
        self._stop = False
        base_url = config.base_url or config.extra.get("vllm_url") or DEFAULT_VLLM_URL
        api_key = config.api_key or config.extra.get("vllm_key") or "dummy"
        model = config.model if config.model else (
            config.extra.get("vllm_model") if config.extra.get("vllm_model") else DEFAULT_VLLM_MODEL
        )
        cwd = config.cwd or config.extra.get("cwd", "") or config.extra.get("working_dir", "")

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        if not model:
            model = await self._discover_model(base_url)
        if not model:
            yield ProviderEvent(type=EventType.ERROR,
                                content="No model specified and could not auto-detect from vLLM server")
            return

        try:
            from openai import AsyncOpenAI
        except ImportError:
            yield ProviderEvent(type=EventType.ERROR,
                                content="openai package required: pip install openai")
            return

        from services.tools.tool_manager import TOOL_DEFINITIONS

        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        messages = self._build_messages(prompt, history, cwd)

        for iteration in range(MAX_TOOL_ITERATIONS):
            if self._stop:
                yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                yield ProviderEvent(type=EventType.DONE)
                return

            try:
                # Non-streaming call to detect tool_calls in the response
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=4096,
                )
            except Exception as e:
                err_str = str(e)
                # If the server doesn't support tools, fall back to plain streaming
                if "tool" in err_str.lower() or "function" in err_str.lower() or "400" in err_str:
                    log.warning("vLLM server doesn't support tools, falling back to plain text: %s", err_str)
                    async for ev in self._plain_stream(client, model, messages):
                        yield ev
                    return
                yield ProviderEvent(type=EventType.ERROR, content=err_str)
                return

            choice = response.choices[0] if response.choices else None
            if not choice:
                yield ProviderEvent(type=EventType.ERROR, content="No response from model")
                return

            msg = choice.message

            # Separate thinking from visible content
            if msg.content:
                thinking, clean = self._extract_thinking(msg.content)
                if thinking:
                    yield ProviderEvent(type=EventType.THINKING, content=thinking)
                if clean:
                    yield ProviderEvent(type=EventType.TEXT, content=clean)

            # Check for tool calls
            if not msg.tool_calls:
                # No more tools — model is done
                break

            # Append the assistant message (with tool_calls) to messages
            messages.append(msg.model_dump())

            # Execute each tool call
            for tc in msg.tool_calls:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="\n⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return

                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}

                # Emit TOOL_START event for frontend
                yield ProviderEvent(
                    type=EventType.TOOL_START,
                    metadata={
                        "title": f"⚙ {fn_name}",
                        "args": json.dumps(fn_args, indent=2),
                    },
                )

                # Execute the tool
                result_text, is_error = await self._execute_tool(fn_name, fn_args, cwd)

                # Emit TOOL_RESULT event for frontend
                yield ProviderEvent(
                    type=EventType.TOOL_RESULT,
                    content=result_text,
                    metadata={"is_error": is_error},
                )

                # Append tool result to messages for the next iteration
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text[:8000],  # truncate for context limits
                })

            # Usage info
            if response.usage:
                yield ProviderEvent(
                    type=EventType.COST,
                    metadata={"usage": {
                        "input_tokens": response.usage.prompt_tokens or 0,
                        "output_tokens": response.usage.completion_tokens or 0,
                    }},
                )
        else:
            yield ProviderEvent(
                type=EventType.TEXT,
                content=f"\n\n⚠ Reached max tool iterations ({MAX_TOOL_ITERATIONS}). Stopping.",
            )

        yield ProviderEvent(type=EventType.DONE)

    # ------------------------------------------------------------------
    # Fallback: plain streaming (no tools)
    # ------------------------------------------------------------------

    async def _plain_stream(self, client, model: str, messages: list[dict]) -> AsyncIterator[ProviderEvent]:
        """Simple streaming text completion — fallback when tools are unsupported."""
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=4096,
            )
            last_chunk = None
            async for chunk in stream:
                if self._stop:
                    yield ProviderEvent(type=EventType.TEXT, content="⏹ *stopped*")
                    yield ProviderEvent(type=EventType.DONE)
                    return
                last_chunk = chunk
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield ProviderEvent(type=EventType.TEXT, content=delta.content)
                if chunk.choices and chunk.choices[0].finish_reason:
                    break

            if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                yield ProviderEvent(
                    type=EventType.COST,
                    metadata={"usage": {
                        "input_tokens": last_chunk.usage.prompt_tokens or 0,
                        "output_tokens": last_chunk.usage.completion_tokens or 0,
                    }},
                )
        except Exception as e:
            yield ProviderEvent(type=EventType.ERROR, content=str(e))
            return

        yield ProviderEvent(type=EventType.DONE)

    async def stop(self):
        self._stop = True

    async def health_check(self):
        base_url = DEFAULT_VLLM_URL
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{base_url}/health")
                return {"status": "ok" if r.status_code == 200 else "error", "provider": self.name}
        except Exception:
            return {"status": "unavailable", "provider": self.name}
