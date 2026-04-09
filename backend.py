"""
CC-UI Backend — FastAPI server for the two-layer task manager UI.

## Layers (frontend architecture)
- Layer 1: Kanban dashboard (index.html) — Running / Done / Stopped columns, bottom chat bar to create tasks
- Layer 2: Task detail — left chat panel (75%) + right git diff panel (25%), opened by clicking a card

## Key concepts
- Tasks run as async background coroutines (_run_claude / _run_qwen / _run_opencode)
- Each task has a session_id enabling follow-up messages to resume the same Claude session
- Tasks persist in tasks.db (SQLite); history is JSON-serialized list of message dicts
- _tasks dict holds in-memory state; _stop flag signals runners to abort

## Task status lifecycle
  created → running → done | stopped | error

## Message history format (stored in task["history"])
  {"role": "user", "content": "..."}                                          # user prompt
  {"role": "assistant", "content": "..."}                                     # text response
  {"role": "assistant", "content": "...", "metadata": {"title": "...", "status": "done|pending"}}  # tool call

## Endpoints
  POST /tasks                  — create + start task
  GET  /tasks                  — list all (summary)
  GET  /tasks/{id}             — full task with history
  POST /tasks/{id}/message     — follow-up to existing session
  POST /tasks/{id}/stop        — signal runner to stop
  DELETE /tasks/{id}           — delete task
  GET  /tasks/{id}/gitdiff     — git status --short + git diff HEAD for task's cwd
  GET  /suggest?path=          — directory autocomplete for cwd input
"""
import asyncio
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    AssistantMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,
)

os.environ.pop("CLAUDECODE", None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cc-ui")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "index.html"))

SKIP_SUBTYPES = {"init", "system"}
QWEN_APPROVAL_MAP = {
    "bypassPermissions": "yolo",
    "acceptEdits": "auto-edit",
    "dontAsk": "yolo",
    "default": "default",
    "plan": "plan",
}

# ── SQLite task store ────────────────────────────────────────────────
DB_PATH = os.path.join(HERE, "tasks.db")

def _db():
    return sqlite3.connect(DB_PATH)

def _init_db():
    with _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            label TEXT, status TEXT, history TEXT,
            session_id TEXT, cwd TEXT, mode TEXT, model TEXT,
            prompt TEXT, created_at TEXT, finished_at TEXT,
            total_cost REAL, usage TEXT
        )""")
        for col, ctype in [("finished_at", "TEXT"), ("total_cost", "REAL"), ("usage", "TEXT")]:
            try:
                con.execute(f"ALTER TABLE tasks ADD COLUMN {col} {ctype}")
                con.commit()
            except Exception:
                pass

_init_db()

# In-memory dict for running tasks (has _stop flag)
_tasks: dict[str, dict] = {}

def _save(task: dict):
    with _db() as con:
        con.execute("""INSERT OR REPLACE INTO tasks
            (id, label, status, history, session_id, cwd, mode, model, prompt, created_at, finished_at, total_cost, usage)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task["id"], task["label"], task["status"],
             json.dumps(task["history"]), task["session_id"],
             task["cwd"], task["mode"], task["model"],
             task["prompt"], task["created_at"], task.get("finished_at"),
             task.get("total_cost", 0), json.dumps(task.get("usage", {}))))

def _load_all() -> dict[str, dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    result = {}
    for r in rows:
        t = {"id": r[0], "label": r[1], "status": r[2],
             "history": json.loads(r[3]), "session_id": r[4],
             "cwd": r[5], "mode": r[6], "model": r[7],
             "prompt": r[8], "created_at": r[9],
             "finished_at": r[10] if len(r) > 10 else None,
             "total_cost": r[11] if len(r) > 11 else 0,
             "usage": json.loads(r[12]) if len(r) > 12 and r[12] else {},
             "_stop": False}
        # Mark interrupted running tasks as stopped
        if t["status"] == "running":
            t["status"] = "stopped"
        result[t["id"]] = t
    return result

_tasks = _load_all()


def _tool_msg(title, content, status="done"):
    return {"role": "assistant", "content": content, "metadata": {"title": title, "status": status}}


def _merge_usage(task: dict, usage: dict | None):
    """Accumulate token usage into the task."""
    if not usage:
        return
    u = task.setdefault("usage", {})
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        if key in usage:
            u[key] = u.get(key, 0) + usage[key]


# ── Claude runner ────────────────────────────────────────────────────
async def _run_claude(task: dict, extra_env: dict | None = None, model: str | None = None):
    tool_calls: list[list] = []
    text = ""
    history = task["history"][:]  # snapshot at start

    def snapshot():
        tools = [_tool_msg(t, c, s) for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text}] if text else []
        return history + tools + cur

    def close_pending():
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    opts = ClaudeAgentOptions(
        permission_mode=task["mode"],
        resume=task["session_id"],
        cwd=task["cwd"] or None,
        env={**os.environ, **(extra_env or {})},
        model=model,
    )

    try:
        async for msg in query(prompt=task["prompt"], options=opts):
            if task["_stop"]:
                for tc in tool_calls:
                    tc[2] = "done"
                task["history"] = snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}]
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return

            if isinstance(msg, AssistantMessage):
                if msg.error:
                    task["history"] = history + [{"role": "assistant", "content": f"❌ {msg.error}"}]
                    task["status"] = "error"
                    task["finished_at"] = datetime.now().isoformat()
                    _save(task)
                    return
                close_pending()
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
                    elif isinstance(block, ThinkingBlock):
                        tool_calls.append(["💭 Thinking", block.thinking[:500], "pending"])
                    elif isinstance(block, ToolUseBlock):
                        args = json.dumps(block.input, indent=2) if block.input else ""
                        tool_calls.append([f"⚙ {block.name}", f"```json\n{args}\n```", "pending"])
                    elif isinstance(block, ToolResultBlock):
                        content = block.content or ""
                        if isinstance(content, list):
                            content = "\n".join(c.get("text", str(c)) for c in content)
                        preview = str(content)[:600] + ("…" if len(str(content)) > 600 else "")
                        close_pending()
                        tool_calls.append([f"{'❌' if block.is_error else '✓'} result", f"```\n{preview}\n```", "done"])
                task["history"] = snapshot()

            elif isinstance(msg, SystemMessage):
                if msg.subtype not in SKIP_SUBTYPES:
                    desc = getattr(msg, "description", None) or msg.subtype
                    tool_calls.append([f"⚙ {desc}", "", "done"])
                    task["history"] = snapshot()

            elif isinstance(msg, ResultMessage):
                task["session_id"] = msg.session_id
                task["total_cost"] = task.get("total_cost", 0) + (msg.total_cost_usd or 0)
                _merge_usage(task, msg.usage)

    except Exception as e:
        task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        return

    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text}] if text else []
    if not tool_calls and not text:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    task["history"] = history + [_tool_msg(t, c, s) for t, c, s in tool_calls] + final
    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)

VLLM_BASE_URL = "http://192.168.170.76:8000"
VLLM_MODEL = "/home/ng6309/datascience/santhosh/models/qwen3.5-9b"


async def _run_vllm(task: dict):
    """Routes to local vLLM server by passing env overrides via ClaudeAgentOptions."""
    await _run_claude(task,
        extra_env={
            "ANTHROPIC_BASE_URL": task.get("vllm_url") or VLLM_BASE_URL,
            "ANTHROPIC_API_KEY": task.get("vllm_key") or "dummy",
        },
        model=task.get("vllm_model") or VLLM_MODEL,
    )


async def _run_qwen(task: dict):
    tool_calls: list[list] = []
    text = ""
    history = task["history"][:]
    approval = QWEN_APPROVAL_MAP.get(task["mode"], "default")

    cmd = ["qwen", "--output-format", "stream-json", "--approval-mode", approval]
    if task["session_id"]:
        cmd += ["--resume", task["session_id"]]

    def snapshot():
        tools = [_tool_msg(t, c, s) for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text}] if text else []
        return history + tools + cur

    def close_pending():
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=task["cwd"] or None,
        )
        proc.stdin.write(task["prompt"].encode())
        await proc.stdin.drain()
        proc.stdin.close()

        async for line in proc.stdout:
            if task["_stop"]:
                proc.terminate()
                for tc in tool_calls:
                    tc[2] = "done"
                task["history"] = snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}]
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
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
                close_pending()
                for block in msg.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "text":
                        text += block.get("text", "")
                    elif btype == "thinking":
                        tool_calls.append(["💭 Thinking", block.get("thinking", "")[:500], "pending"])
                    elif btype == "tool_use":
                        args = json.dumps(block.get("input", {}), indent=2)
                        tool_calls.append([f"⚙ {block.get('name', '')}", f"```json\n{args}\n```", "pending"])
                    elif btype == "tool_result":
                        content = block.get("content", "")
                        if isinstance(content, list):
                            content = "\n".join(c.get("text", str(c)) for c in content)
                        preview = str(content)[:600] + ("…" if len(str(content)) > 600 else "")
                        close_pending()
                        tool_calls.append([f"{'❌' if block.get('is_error') else '✓'} result", f"```\n{preview}\n```", "done"])
                task["history"] = snapshot()
            elif mtype == "system" and msg.get("subtype") not in SKIP_SUBTYPES:
                tool_calls.append([f"⚙ {msg.get('subtype', 'system')}", "", "done"])
                task["history"] = snapshot()
            elif mtype == "result":
                task["session_id"] = msg.get("session_id")

        await proc.wait()
    except Exception as e:
        task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        return

    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text}] if text else []
    if not tool_calls and not text:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    task["history"] = history + [_tool_msg(t, c, s) for t, c, s in tool_calls] + final
    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)


async def _run_opencode(task: dict):
    tool_calls: list[list] = []
    text = ""
    history = task["history"][:]
    session_id = None

    cmd = ["opencode", "run", "--format", "json"]
    if task["session_id"]:
        cmd += ["--session", task["session_id"]]
    if task["cwd"]:
        cmd += ["--dir", task["cwd"]]
    cmd += [task["prompt"]]

    def snapshot():
        tools = [_tool_msg(t, c, s) for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text}] if text else []
        return history + tools + cur

    def close_pending():
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async for line in proc.stdout:
            if task["_stop"]:
                proc.terminate()
                for tc in tool_calls:
                    tc[2] = "done"
                task["history"] = snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}]
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return

            line = line.decode().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue

            # Capture session_id from any event
            if msg.get("sessionID"):
                session_id = msg["sessionID"]

            mtype = msg.get("type")
            part = msg.get("part", {})

            if mtype == "text":
                text += part.get("text", "")
                task["history"] = snapshot()
            elif mtype == "tool_use":
                # opencode sends one event per tool call with state.input + state.output
                state = part.get("state", {})
                tool_name = part.get("tool", "tool")
                title = state.get("title") or tool_name
                args = json.dumps(state.get("input", {}), indent=2)
                output = state.get("output", "")
                if isinstance(output, list):
                    output = "\n".join(c.get("text", str(c)) for c in output)
                preview = str(output)[:600] + ("…" if len(str(output)) > 600 else "")
                is_error = state.get("metadata", {}).get("exit", 0) != 0
                content = f"```json\n{args}\n```\n**Output:**\n```\n{preview}\n```"
                tool_calls.append([f"{'❌' if is_error else '⚙'} {title}", content, "done"])
                task["history"] = snapshot()

        await proc.wait()
    except Exception as e:
        task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        return

    if session_id:
        task["session_id"] = session_id
    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text}] if text else []
    if not tool_calls and not text:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    task["history"] = history + [_tool_msg(t, c, s) for t, c, s in tool_calls] + final
    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)


# ── Claude Multi-Agent runner ───────────────────────────────────────

_SUPERVISOR_PLAN_TMPL = (
    "You are a task supervisor. Given the user's request, break it into independent "
    "subtasks that can be executed in parallel by separate Claude agents.\n\n"
    'IMPORTANT: Respond ONLY with a JSON array. No markdown, no explanation, just the JSON.\n'
    'Each subtask: {{"id": "1", "prompt": "specific instruction", "model": "haiku"}}\n'
    'Use "haiku" for most tasks (fast/cheap). Use "sonnet" only for complex reasoning.\n'
    "Keep subtasks independent — no cross-dependencies.\n"
    "If the task is simple enough for one agent, return a single-element array.\n\n"
    "User request: {prompt}\n"
    "Working directory: {cwd}"
)

_SUPERVISOR_REVIEW_TMPL = (
    "You are a task supervisor reviewing agent results.\n\n"
    "Original request: {prompt}\n"
    "Working directory: {cwd}\n\n"
    "Agent results:\n{results}\n\n"
    "Decide:\n"
    '1. If all work is complete and correct, respond with EXACTLY: {{"done": true, "summary": "brief summary of what was accomplished"}}\n'
    '2. If more work is needed, respond with EXACTLY: {{"done": false, "tasks": [{{"id": "1", "prompt": "what to do next", "model": "haiku"}}]}}\n\n'
    "Respond ONLY with JSON. No markdown, no explanation."
)


async def _run_agent_worker(agent_id: str, prompt: str, cwd: str, mode: str,
                            model: str = "claude-haiku-4-5-20251001",
                            parent_task: dict = None) -> dict:
    """Run a single agent worker and collect its history."""
    worker = {
        "id": str(uuid.uuid4()),
        "label": f"agent-{agent_id}",
        "status": "running",
        "history": [{"role": "user", "content": prompt}],
        "session_id": None,
        "cwd": cwd,
        "mode": mode,
        "model": "claude",
        "prompt": prompt,
        "created_at": datetime.now().isoformat(),
        "_stop": False,
    }

    # Link stop flag to parent
    original_stop = worker["_stop"]

    tool_calls: list[list] = []
    text = ""
    history = worker["history"][:]

    def snapshot():
        tools = [_tool_msg(t, c, s) for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text}] if text else []
        return history + tools + cur

    def close_pending():
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    opts = ClaudeAgentOptions(
        permission_mode=mode,
        cwd=cwd or None,
        model=model,
    )

    try:
        async for msg in query(prompt=prompt, options=opts):
            if parent_task and parent_task["_stop"]:
                worker["_stop"] = True
                for tc in tool_calls:
                    tc[2] = "done"
                worker["history"] = snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}]
                worker["status"] = "stopped"
                return worker

            if isinstance(msg, AssistantMessage):
                if msg.error:
                    worker["history"] = history + [{"role": "assistant", "content": f"❌ {msg.error}"}]
                    worker["status"] = "error"
                    return worker
                close_pending()
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
                    elif isinstance(block, ThinkingBlock):
                        tool_calls.append(["💭 Thinking", block.thinking[:500], "pending"])
                    elif isinstance(block, ToolUseBlock):
                        args = json.dumps(block.input, indent=2) if block.input else ""
                        tool_calls.append([f"⚙ {block.name}", f"```json\n{args}\n```", "pending"])
                    elif isinstance(block, ToolResultBlock):
                        content = block.content or ""
                        if isinstance(content, list):
                            content = "\n".join(c.get("text", str(c)) for c in content)
                        preview = str(content)[:600] + ("…" if len(str(content)) > 600 else "")
                        close_pending()
                        tool_calls.append([f"{'❌' if block.is_error else '✓'} result", f"```\n{preview}\n```", "done"])

                # Update parent task history in real-time
                if parent_task:
                    _update_agent_group(parent_task, agent_id, model, snapshot())
                    _save(parent_task)

            elif isinstance(msg, ResultMessage):
                worker["session_id"] = msg.session_id
                if parent_task:
                    parent_task["total_cost"] = parent_task.get("total_cost", 0) + (msg.total_cost_usd or 0)
                    _merge_usage(parent_task, msg.usage)

    except Exception as e:
        worker["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        worker["status"] = "error"
        return worker

    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text}] if text else []
    if not tool_calls and not text:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    worker["history"] = history + [_tool_msg(t, c, s) for t, c, s in tool_calls] + final
    worker["status"] = "done"
    return worker


def _update_agent_group(task: dict, agent_id: str, model: str, agent_history: list):
    """Update or insert an agent-group entry in the parent task's history."""
    # Find existing group for this agent_id
    for entry in task["history"]:
        if entry.get("role") == "agent-group" and entry.get("agent_id") == agent_id:
            entry["children"] = agent_history[1:]  # skip the user prompt
            return
    # Insert new group before any final summary
    task["history"].append({
        "role": "agent-group",
        "agent_id": agent_id,
        "agent_label": f"Agent {agent_id} ({model.split('-')[1] if '-' in model else model})",
        "model": model,
        "status": "running",
        "children": agent_history[1:],
    })


async def _supervisor_call(prompt: str, cwd: str, mode: str = "bypassPermissions",
                           parent_task: dict | None = None) -> str:
    """Quick single-turn call to sonnet for planning/reviewing."""
    log.info("[supervisor] calling sonnet, prompt[:100]=%s", prompt[:100])
    text = ""
    opts = ClaudeAgentOptions(
        permission_mode=mode,
        cwd=cwd or None,
        model="claude-sonnet-4-6",
    )
    try:
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
            elif isinstance(msg, ResultMessage) and parent_task:
                parent_task["total_cost"] = parent_task.get("total_cost", 0) + (msg.total_cost_usd or 0)
                _merge_usage(parent_task, msg.usage)
    except Exception as e:
        log.error("[supervisor] query error: %s", e, exc_info=True)
        raise
    log.info("[supervisor] response len=%d, first 200 chars: %s", len(text), repr(text[:200]))
    return text


async def _run_claudeagents(task: dict):
    """Multi-agent supervisor loop: plan → parallel workers → review → loop."""
    log.info("[claudeagents] START task=%s prompt=%s", task["id"][:8], task["prompt"][:80])
    cwd = task["cwd"] or os.getcwd()
    mode = task["mode"]
    user_prompt = task["prompt"]
    base_history = task["history"][:]  # starts with [user msg]

    try:
        # ── Step 1: Supervisor plans subtasks ──
        task["history"] = base_history + [
            _tool_msg("🎯 Supervisor", "Planning subtasks…", "pending")
        ]
        _save(task)

        plan_prompt = _SUPERVISOR_PLAN_TMPL.format(prompt=user_prompt, cwd=cwd)
        plan_raw = await _supervisor_call(plan_prompt, cwd, mode, parent_task=task)
        log.info("[claudeagents] plan_raw len=%d", len(plan_raw))

        # Parse JSON from response (strip markdown fences if any)
        plan_text = plan_raw.strip()
        if plan_text.startswith("```"):
            plan_text = "\n".join(plan_text.split("\n")[1:])
            if plan_text.endswith("```"):
                plan_text = plan_text[:-3]

        try:
            parsed = json.loads(plan_text.strip())
            log.info("[claudeagents] direct JSON parse OK")
        except json.JSONDecodeError as je:
            log.warning("[claudeagents] direct JSON parse failed: %s", je)
            # Try to extract JSON array from the response
            match = re.search(r'\[.*\]', plan_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                log.info("[claudeagents] regex JSON parse OK")
            else:
                log.warning("[claudeagents] regex fallback failed, using single task")
                # Fallback: single task with the original prompt
                parsed = [{"id": "1", "prompt": user_prompt, "model": "haiku"}]

        # Normalize: could be a bare array or {"tasks": [...]}
        if isinstance(parsed, dict):
            subtasks = parsed.get("tasks", [parsed])
        else:
            subtasks = parsed

        # Ensure each subtask has an id and prompt
        for i, st in enumerate(subtasks):
            if "id" not in st:
                st["id"] = str(i + 1)
            if "prompt" not in st:
                st["prompt"] = st.get("task", st.get("description", user_prompt))

        log.info("[claudeagents] %d subtasks: %s", len(subtasks), [s["id"] for s in subtasks])

        task["history"] = base_history + [
            _tool_msg("🎯 Supervisor", f"Planned **{len(subtasks)}** subtask(s):\n" +
                      "\n".join(f"- **{s['id']}**: {s['prompt'][:80]}" for s in subtasks), "done")
        ]
        _save(task)

        MODEL_MAP = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
        }

        max_rounds = 5
        for round_num in range(max_rounds):
            if task["_stop"]:
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return

            # ── Run workers in parallel ──
            log.info("[claudeagents] round %d: dispatching %d workers", round_num + 1, len(subtasks))
            workers = []
            for st in subtasks:
                model_key = st.get("model", "haiku")
                model = MODEL_MAP.get(model_key, MODEL_MAP["haiku"])
                log.info("[claudeagents]   worker id=%s model=%s prompt=%s", st["id"], model, st["prompt"][:60])
                workers.append(
                    _run_agent_worker(st["id"], st["prompt"], cwd, mode, model, task)
                )

            results = await asyncio.gather(*workers)
            log.info("[claudeagents] round %d: workers done, statuses=%s", round_num + 1, [r["status"] for r in results])

            # Mark agent groups as done
            for entry in task["history"]:
                if entry.get("role") == "agent-group":
                    matching = [r for r in results if r["label"] == f"agent-{entry['agent_id']}"]
                    if matching:
                        entry["status"] = matching[0]["status"]

            _save(task)

            # ── Supervisor reviews ──
            task["history"].append(
                _tool_msg("🎯 Supervisor", f"Reviewing round {round_num + 1} results…", "pending")
            )
            _save(task)

            results_summary = ""
            for r in results:
                last_text = ""
                for msg in reversed(r["history"]):
                    if msg.get("role") == "assistant" and not msg.get("metadata"):
                        last_text = str(msg.get("content", ""))[:300]
                        break
                results_summary += f"\n--- Agent {r['label']} ({r['status']}) ---\n{last_text}\n"

            review_prompt = _SUPERVISOR_REVIEW_TMPL.format(
                prompt=user_prompt, cwd=cwd, results=results_summary
            )
            review_raw = await _supervisor_call(review_prompt, cwd, mode, parent_task=task)

            review_text = review_raw.strip()
            if review_text.startswith("```"):
                review_text = "\n".join(review_text.split("\n")[1:])
                if review_text.endswith("```"):
                    review_text = review_text[:-3]

            try:
                review = json.loads(review_text.strip())
            except json.JSONDecodeError:
                # If can't parse, assume done
                review = {"done": True, "summary": review_raw[:200]}

            if review.get("done"):
                # Update the pending review tool msg
                for entry in reversed(task["history"]):
                    if entry.get("metadata", {}).get("status") == "pending":
                        entry["metadata"]["status"] = "done"
                        entry["content"] = f"All done ✓ — {review.get('summary', 'Complete')}"
                        break
                task["history"].append({
                    "role": "assistant",
                    "content": review.get("summary", "All tasks completed successfully.")
                })
                task["status"] = "done"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return
            else:
                # More work needed — update review msg and loop
                for entry in reversed(task["history"]):
                    if entry.get("metadata", {}).get("status") == "pending":
                        entry["metadata"]["status"] = "done"
                        new_tasks = review.get("tasks", [])
                        entry["content"] = f"Round {round_num + 1} done. Dispatching **{len(new_tasks)}** more task(s)…"
                        break
                subtasks = review.get("tasks", [])
                _save(task)

        # Max rounds reached
        task["history"].append({"role": "assistant", "content": "⚠ Max rounds reached. Stopping."})
        task["status"] = "done"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)

    except Exception as e:
        log.error("[claudeagents] EXCEPTION: %s", e, exc_info=True)
        task["history"] = task["history"] + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)


# ── API models ───────────────────────────────────────────────────────
class CreateTaskRequest(BaseModel):
    prompt: str
    mode: str = "bypassPermissions"
    model: str = "claude"
    cwd: str = ""
    session_id: str | None = None
    vllm_url: str | None = None
    vllm_key: str | None = None
    vllm_model: str | None = None


class SendMessageRequest(BaseModel):
    prompt: str


# ── API routes ───────────────────────────────────────────────────────
@app.post("/tasks")
async def create_task(req: CreateTaskRequest):
    task_id = str(uuid.uuid4())
    label = req.prompt[:40].strip() + ("…" if len(req.prompt) > 40 else "")
    task = {
        "id": task_id,
        "label": label,
        "status": "running",
        "history": [{"role": "user", "content": req.prompt}],
        "session_id": req.session_id,
        "cwd": req.cwd or os.getcwd(),
        "mode": req.mode,
        "model": req.model,
        "prompt": req.prompt,
        "created_at": datetime.now().isoformat(),
        "_stop": False,
        "total_cost": 0,
        "usage": {},
        "vllm_url": req.vllm_url,
        "vllm_key": req.vllm_key,
        "vllm_model": req.vllm_model,
    }
    _tasks[task_id] = task
    _save(task)

    # Fire and forget — runs in background
    if req.model == "qwen":
        runner = _run_qwen(task)
    elif req.model == "opencode":
        runner = _run_opencode(task)
    elif req.model == "vllm":
        runner = _run_vllm(task)
    elif req.model == "claudeagents":
        runner = _run_claudeagents(task)
    else:
        runner = _run_claude(task)
    asyncio.create_task(runner)

    return {"id": task_id, "label": label, "status": "running"}


@app.post("/tasks/{task_id}/message")
async def send_message(task_id: str, req: SendMessageRequest):
    """Send a follow-up message to an existing task session."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] == "running":
        raise HTTPException(409, "Task is still running")

    task["history"] = task["history"] + [{"role": "user", "content": req.prompt}]
    task["prompt"] = req.prompt
    task["status"] = "running"
    task["_stop"] = False

    if task["model"] == "qwen":
        runner = _run_qwen(task)
    elif task["model"] == "opencode":
        runner = _run_opencode(task)
    elif task["model"] == "vllm":
        runner = _run_vllm(task)
    elif task["model"] == "claudeagents":
        runner = _run_claudeagents(task)
    else:
        runner = _run_claude(task)
    asyncio.create_task(runner)

    return {"id": task_id, "status": "running"}


@app.get("/suggest")
async def suggest_path(path: str = ""):
    try:
        path = os.path.expanduser(path) if path else ""
        if not path:
            return []

        # Exact dir typed — list its children instantly
        if os.path.isdir(path):
            results = [
                os.path.join(path, e) for e in sorted(os.listdir(path))
                if not e.startswith(".") and os.path.isdir(os.path.join(path, e))
            ]
            return results[:15]

        # Partial path: match in parent dir first (instant), then broader search
        search_dir = os.path.dirname(path) or os.path.expanduser("~")
        pattern = os.path.basename(path).lower()

        # Fast: check parent dir children
        if os.path.isdir(search_dir):
            quick = [
                os.path.join(search_dir, e) for e in sorted(os.listdir(search_dir))
                if not e.startswith(".") and os.path.isdir(os.path.join(search_dir, e))
                and pattern in e.lower()
            ]
            if quick:
                return quick[:15]

        # Fallback: async find from home, max 3 levels, 1s timeout
        home = os.path.expanduser("~")
        proc = await asyncio.create_subprocess_exec(
            "find", home, "-maxdepth", "3", "-type", "d",
            "-iname", f"*{pattern}*", "-not", "-path", "*/.*",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
        except asyncio.TimeoutError:
            proc.kill()
            stdout = b""
        results = [l for l in stdout.decode().splitlines() if "/." not in l]
        return results[:15]
    except Exception:
        return []


@app.get("/tasks")
async def list_tasks():
    result = []
    for t in sorted(_tasks.values(), key=lambda x: x["created_at"], reverse=True):
        last = ""
        for msg in reversed(t["history"]):
            if msg.get("role") == "assistant" and not msg.get("metadata"):
                last = str(msg.get("content", ""))[:60]
                break
        result.append({
            "id": t["id"],
            "label": t["label"],
            "status": t["status"],
            "created_at": t["created_at"],
            "finished_at": t.get("finished_at"),
            "preview": last,
            "cwd": t["cwd"],
            "model": t.get("model"),
            "mode": t.get("mode"),
            "total_cost": t.get("total_cost", 0),
            "usage": t.get("usage", {}),
        })
    return result


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "id": task["id"],
        "label": task["label"],
        "status": task["status"],
        "history": task["history"],
        "session_id": task["session_id"],
        "cwd": task["cwd"],
        "created_at": task["created_at"],
        "total_cost": task.get("total_cost", 0),
        "usage": task.get("usage", {}),
    }


@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_stop"] = True
    return {"status": "stopping"}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    task = _tasks.pop(task_id, None)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_stop"] = True
    with _db() as con:
        con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    return {"status": "deleted"}


@app.get("/tasks/{task_id}/gitdiff")
async def get_gitdiff(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    cwd = task["cwd"] or os.getcwd()
    try:
        # Check if git repo
        p = await asyncio.create_subprocess_exec(
            "git", "-C", cwd, "rev-parse", "--is-inside-work-tree",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        await p.communicate()
        if p.returncode != 0:
            return {"is_git": False, "status": "", "diff": ""}

        # git status --short
        p1 = await asyncio.create_subprocess_exec(
            "git", "-C", cwd, "status", "--short",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        s_out, _ = await p1.communicate()

        # git diff HEAD
        p2 = await asyncio.create_subprocess_exec(
            "git", "-C", cwd, "diff", "HEAD",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        d_out, _ = await p2.communicate()

        return {
            "is_git": True,
            "status": s_out.decode(errors="replace"),
            "diff": d_out.decode(errors="replace"),
        }
    except Exception:
        return {"is_git": False, "status": "", "diff": ""}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
