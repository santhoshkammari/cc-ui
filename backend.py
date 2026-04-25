"""
CC-UI v2 Backend — Unified FastAPI gateway.

Integrates services:
- Provider registry (Claude Code, OpenCode, Copilot, Local/vLLM)
- Task management with unified provider interface
- Scheduler (cron/delayed/recurring jobs)
- Monitor (health, metrics, system info)
- Git service (branches, commits, diffs)
- Directory autocomplete
"""
import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.providers.base import ProviderConfig, ProviderEvent, EventType
from services.providers.registry import get_provider, list_providers, health_check_all
from services.scheduler import Scheduler
from services.monitor import Monitor
from services.git_service import GitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cc-ui")

app = FastAPI(title="CC-UI v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Database ─────────────────────────────────────────────────────────
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
            total_cost REAL, usage TEXT, branch TEXT
        )""")
        for col, ctype in [("finished_at","TEXT"),("total_cost","REAL"),("usage","TEXT"),("branch","TEXT"),("advisor","TEXT"),("advisor_model","TEXT"),("deleted_at","TEXT")]:
            try:
                con.execute(f"ALTER TABLE tasks ADD COLUMN {col} {ctype}")
                con.commit()
            except Exception:
                pass
        con.execute("""CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT UNIQUE NOT NULL
        )""")

_init_db()

# ── In-memory state ──────────────────────────────────────────────────
_tasks: dict[str, dict] = {}
_providers: dict[str, Any] = {}  # active provider instances per task

# ── Services ─────────────────────────────────────────────────────────
monitor = Monitor()


async def _create_task_callback(prompt, model, mode, cwd):
    """Callback for scheduler to create tasks."""
    task_id = str(uuid.uuid4())
    label = prompt[:40].strip() + ("…" if len(prompt) > 40 else "")
    task = {
        "id": task_id, "label": label, "status": "running",
        "history": [{"role": "user", "content": prompt}],
        "session_id": None, "cwd": cwd or os.getcwd(),
        "mode": mode, "model": model, "prompt": prompt,
        "created_at": datetime.now().isoformat(), "_stop": False,
        "total_cost": 0, "usage": {}, "branch": "",
    }
    _tasks[task_id] = task
    _save(task)
    asyncio.create_task(_run_task(task))
    return task_id


scheduler = Scheduler(DB_PATH, task_callback=_create_task_callback)


# ── Task persistence ─────────────────────────────────────────────────
def _save(task: dict):
    if task.get("_deleted"):
        return
    with _db() as con:
        con.execute("""INSERT OR REPLACE INTO tasks
            (id, label, status, history, session_id, cwd, mode, model, prompt, created_at, finished_at, total_cost, usage, branch, advisor, advisor_model, deleted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task["id"], task["label"], task["status"],
             json.dumps(task["history"]), task["session_id"],
             task["cwd"], task["mode"], task["model"],
             task["prompt"], task["created_at"], task.get("finished_at"),
             task.get("total_cost", 0), json.dumps(task.get("usage", {})),
             task.get("branch", ""), task.get("advisor", ""), task.get("advisor_model", ""),
             task.get("deleted_at")))

def _load_all() -> dict[str, dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    result = {}
    cols = ["id","label","status","history","session_id","cwd","mode","model","prompt","created_at","finished_at","total_cost","usage","branch","advisor","advisor_model","deleted_at"]
    for r in rows:
        t = {}
        for i, col in enumerate(cols):
            if i < len(r):
                t[col] = r[i]
            else:
                t[col] = None
        t["history"] = json.loads(t["history"] or "[]")
        t["usage"] = json.loads(t["usage"] or "{}")
        t["total_cost"] = t["total_cost"] or 0
        t["branch"] = t.get("branch") or ""
        t["advisor"] = t.get("advisor") or ""
        t["advisor_model"] = t.get("advisor_model") or ""
        t["_stop"] = False
        if t["status"] == "running":
            t["status"] = "stopped"
        result[t["id"]] = t
    return result

_tasks = _load_all()


# ── Unified task runner ──────────────────────────────────────────────
def _merge_usage(task: dict, usage: dict | None):
    if not usage:
        return
    u = task.setdefault("usage", {})
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        if key in usage:
            u[key] = u.get(key, 0) + usage[key]


async def _run_task(task: dict):
    """Run a task using the unified provider interface."""
    model_name = task["model"]

    # Map agent IDs to provider names for routing
    _AGENT_TO_PROVIDER = {
        "claude-code": "claude",
        "opencode":    "opencode",
        "copilot":     "copilot",
        "local":       "vllm",
    }

    # Map short model names to specific Claude model IDs
    _MODEL_ALIASES = {
        "sonnet": "claude-sonnet-4-20250514",
        "opus":   "claude-opus-4-20250514",
        "haiku":  "claude-haiku-4-5-20251001",
    }

    # Resolve agent ID to provider name
    provider_name = _AGENT_TO_PROVIDER.get(model_name, model_name)

    if model_name in _MODEL_ALIASES:
        task.setdefault("extra", {})["model_override"] = _MODEL_ALIASES[model_name]

    # For "local" agent, inject vLLM defaults if not in extra
    if model_name == "local":
        extra = task.setdefault("extra", {})
        if not extra.get("vllm_url"):
            extra["vllm_url"] = "http://192.168.170.76:8000"
        # Empty model string lets vLLM auto-pick the loaded model
        if "vllm_model" not in extra:
            extra["vllm_model"] = ""

    log.info("_run_task starting: agent=%s provider=%s prompt=%s", model_name, provider_name, task["prompt"][:60])
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        log.error("_run_task: provider not found: %s", e)
        task["history"].append({"role": "assistant", "content": f"❌ {e}"})
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        return

    _providers[task["id"]] = provider

    config = ProviderConfig(
        model=task.get("extra", {}).get("vllm_model") or task.get("extra", {}).get("model_override") or "",
        mode=task["mode"],
        cwd=task["cwd"],
        session_id=task["session_id"],
        base_url=task.get("extra", {}).get("vllm_url") or task.get("extra", {}).get("base_url") or "",
        api_key=task.get("extra", {}).get("vllm_key") or task.get("extra", {}).get("api_key") or "",
        extra=task.get("extra", {}),
    )

    history_snapshot = task["history"][:]
    tool_calls: list[list] = []  # [title, content, status]
    agent_groups: list[dict] = []  # agent group entries
    text_buf = ""

    def snapshot():
        tools = [{"role": "assistant", "content": c, "metadata": {"title": t, "status": s}} for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text_buf}] if text_buf else []
        return history_snapshot + agent_groups + tools + cur

    def close_pending():
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    try:
        async for event in provider.run(task["prompt"], config, task["history"]):
            if task["_stop"]:
                await provider.stop()
                for tc in tool_calls:
                    tc[2] = "done"
                task["history"] = snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}]
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return

            if event.type == EventType.TEXT:
                text_buf += event.content
                task["history"] = snapshot()

            elif event.type == EventType.TOOL_START:
                close_pending()
                title = event.metadata.get("title", "⚙ tool")
                args = event.metadata.get("args", "")
                tool_calls.append([title, f"```json\n{args}\n```" if args else "", "pending"])
                task["history"] = snapshot()

            elif event.type == EventType.TOOL_RESULT:
                close_pending()
                is_error = event.metadata.get("is_error", False)
                icon = "❌" if is_error else "✓"
                preview = event.content[:600] + ("…" if len(event.content) > 600 else "")
                tool_calls.append([f"{icon} result", f"```\n{preview}\n```", "done"])
                task["history"] = snapshot()

            elif event.type == EventType.THINKING:
                tool_calls.append(["💭 Thinking", event.content[:500], "done"])
                task["history"] = snapshot()

            elif event.type == EventType.AGENT_GROUP:
                agent_id = event.metadata.get("agent_id", "")
                found = False
                for entry in agent_groups:
                    if entry.get("agent_id") == agent_id:
                        entry.update(event.metadata)
                        entry["role"] = "agent-group"
                        found = True
                        break
                if not found:
                    entry = {"role": "agent-group"}
                    entry.update(event.metadata)
                    agent_groups.append(entry)
                task["history"] = snapshot()

            elif event.type == EventType.COST:
                if event.metadata.get("session_id"):
                    task["session_id"] = event.metadata["session_id"]
                if event.metadata.get("total_cost_usd"):
                    task["total_cost"] = task.get("total_cost", 0) + event.metadata["total_cost_usd"]
                _merge_usage(task, event.metadata.get("usage"))

            elif event.type == EventType.ERROR:
                task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {event.content}"}]
                task["status"] = "error"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                _providers.pop(task["id"], None)
                return

            elif event.type == EventType.DONE:
                break

            _save(task)

    except Exception as e:
        log.exception("_run_task exception: %s", e)
        task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        _providers.pop(task["id"], None)
        return

    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text_buf}] if text_buf else []
    if not tool_calls and not text_buf and not agent_groups:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    log.info("_run_task done: text_buf=%s tools=%d agents=%d", repr(text_buf)[:80], len(tool_calls), len(agent_groups))
    task["history"] = history_snapshot + agent_groups + [
        {"role": "assistant", "content": c, "metadata": {"title": t, "status": s}}
        for t, c, s in tool_calls
    ] + final

    # ── Advisor review loop ──────────────────────────────────────────
    advisor_name = task.get("advisor", "")
    if advisor_name and text_buf and not task["_stop"]:
        await _run_advisor_review(task, text_buf)
        return  # advisor sets status and saves

    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)
    _providers.pop(task["id"], None)


async def _run_advisor_review(task: dict, worker_output: str, max_rounds: int = 2):
    """Run advisor review on worker output. Advisor reviews and optionally triggers worker refinement."""
    advisor_name = task.get("advisor", "")
    advisor_model = task.get("advisor_model", "")
    log.info("Advisor review: advisor=%s model=%s", advisor_name, advisor_model)

    try:
        advisor_provider = get_provider(advisor_name)
    except ValueError as e:
        log.warning("Advisor provider not found: %s — skipping review", e)
        task["status"] = "done"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        _providers.pop(task["id"], None)
        return

    advisor_config = ProviderConfig(
        model=advisor_model,
        mode="plan",
        cwd=task["cwd"],
        session_id=None,
        base_url=task.get("extra", {}).get("base_url", ""),
        api_key=task.get("extra", {}).get("api_key", ""),
        extra=task.get("extra", {}),
    )

    review_prompt = f"""You are an advisor reviewing work done by another AI agent.

**Original task:** {task['prompt']}

**Worker's output:**
{worker_output[:4000]}

Please review this output. Provide:
1. A brief assessment (is it correct, complete, well-structured?)
2. Any issues or improvements needed
3. A final verdict: APPROVE if the work is good enough, or REVISE if it needs changes.

Keep your review concise and actionable."""

    # Add advisor marker to history
    task["history"].append({
        "role": "agent-group",
        "agent_id": "advisor",
        "agent_name": f"🧠 Advisor ({advisor_name})",
        "status": "running",
        "children": [],
    })
    _save(task)

    # Find the advisor agent-group entry
    advisor_entry = None
    for entry in task["history"]:
        if entry.get("role") == "agent-group" and entry.get("agent_id") == "advisor":
            advisor_entry = entry
            break

    advisor_text = ""
    try:
        async for event in advisor_provider.run(review_prompt, advisor_config, []):
            if task["_stop"]:
                await advisor_provider.stop()
                if advisor_entry:
                    advisor_entry["status"] = "stopped"
                task["status"] = "stopped"
                task["finished_at"] = datetime.now().isoformat()
                _save(task)
                return

            if event.type == EventType.TEXT:
                advisor_text += event.content
                if advisor_entry:
                    advisor_entry["children"] = [{"role": "assistant", "content": advisor_text}]
                    _save(task)
            elif event.type == EventType.COST:
                if event.metadata.get("total_cost_usd"):
                    task["total_cost"] = task.get("total_cost", 0) + event.metadata["total_cost_usd"]
                _merge_usage(task, event.metadata.get("usage"))
            elif event.type == EventType.DONE:
                break

    except Exception as e:
        log.exception("Advisor review failed: %s", e)
        if advisor_entry:
            advisor_entry["status"] = "error"
            advisor_entry["children"] = [{"role": "assistant", "content": f"❌ Advisor error: {e}"}]

    if advisor_entry:
        advisor_entry["status"] = "done"
        if not advisor_entry.get("children"):
            advisor_entry["children"] = [{"role": "assistant", "content": advisor_text or "*(no review)*"}]

    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)
    _providers.pop(task["id"], None)


# ── API Models ───────────────────────────────────────────────────────
class CreateTaskRequest(BaseModel):
    prompt: str
    model: str = "claude-code"
    agent_id: str = ""             # which agent to use (e.g. "claude-code", "chat")
    mode: str = "bypassPermissions"  # kept for backward compat
    cwd: str = ""
    session_id: str | None = None
    branch: str = ""
    extra: dict = {}
    advisor: str = ""              # deprecated — kept for backward compat
    advisor_model: str = ""        # deprecated — kept for backward compat

class SendMessageRequest(BaseModel):
    prompt: str

class WorkspaceRequest(BaseModel):
    path: str
    name: str = ""

class GitStageRequest(BaseModel):
    files: list[str]

class CreateJobRequest(BaseModel):
    name: str
    prompt: str
    model: str = "claude-code"
    mode: str = "bypassPermissions"
    cwd: str = ""
    schedule: str = ""
    interval_seconds: int = 0
    delay_seconds: int = 0
    one_shot: bool = False

class GitBranchRequest(BaseModel):
    name: str
    checkout: bool = True

class GitCommitRequest(BaseModel):
    message: str
    add_all: bool = True

class GitPRCreateRequest(BaseModel):
    title: str
    body: str = ""
    base: str = "main"
    head: str = ""

class GitPRMergeRequest(BaseModel):
    pr_number: int
    method: str = "merge"  # merge, squash, rebase


# ── Routes: Static ───────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "index.html"))


# ── Routes: Tasks ────────────────────────────────────────────────────
@app.post("/tasks")
async def create_task(req: CreateTaskRequest):
    task_id = str(uuid.uuid4())
    label = req.prompt[:40].strip() + ("…" if len(req.prompt) > 40 else "")

    # Handle branch creation
    branch = req.branch
    if branch and req.cwd:
        result = await GitService.create_branch(req.cwd, branch)
        if not result.get("success"):
            # Try switching instead
            await GitService.switch_branch(req.cwd, branch)

    task = {
        "id": task_id, "label": label, "status": "running",
        "history": [{"role": "user", "content": req.prompt}],
        "session_id": req.session_id, "cwd": req.cwd or os.getcwd(),
        "mode": req.mode, "model": req.model, "prompt": req.prompt,
        "created_at": datetime.now().isoformat(), "_stop": False,
        "total_cost": 0, "usage": {}, "branch": branch,
        "extra": req.extra,
        "advisor": req.advisor, "advisor_model": req.advisor_model,
    }
    _tasks[task_id] = task
    _save(task)
    asyncio.create_task(_run_task(task))
    return {"id": task_id, "label": label, "status": "running"}


@app.get("/tasks")
async def list_tasks_route():
    monitor.update_task_stats(_tasks)
    result = []
    for t in sorted(_tasks.values(), key=lambda x: x["created_at"], reverse=True):
        if t.get("deleted_at"):
            continue
        last = ""
        for msg in reversed(t["history"]):
            if msg.get("role") == "assistant" and not msg.get("metadata"):
                last = str(msg.get("content", ""))[:60]
                break
        result.append({
            "id": t["id"], "label": t["label"], "status": t["status"],
            "created_at": t["created_at"], "finished_at": t.get("finished_at"),
            "preview": last, "cwd": t["cwd"], "model": t.get("model"),
            "mode": t.get("mode"), "total_cost": t.get("total_cost", 0),
            "usage": t.get("usage", {}), "branch": t.get("branch", ""),
            "advisor": t.get("advisor", ""), "advisor_model": t.get("advisor_model", ""),
        })
    return result


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "id": task["id"], "label": task["label"], "status": task["status"],
        "history": task["history"], "session_id": task["session_id"],
        "cwd": task["cwd"], "created_at": task["created_at"],
        "total_cost": task.get("total_cost", 0),
        "usage": task.get("usage", {}), "branch": task.get("branch", ""),
        "advisor": task.get("advisor", ""), "advisor_model": task.get("advisor_model", ""),
    }


@app.post("/tasks/{task_id}/message")
async def send_message(task_id: str, req: SendMessageRequest):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] == "running":
        raise HTTPException(409, "Task is still running")

    task["history"] = task["history"] + [{"role": "user", "content": req.prompt}]
    task["prompt"] = req.prompt
    task["status"] = "running"
    task["_stop"] = False
    asyncio.create_task(_run_task(task))
    return {"id": task_id, "status": "running"}


@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_stop"] = True
    provider = _providers.get(task_id)
    if provider:
        await provider.stop()
    return {"status": "stopping"}


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] == "running":
        raise HTTPException(409, "Task is already running")
    resume_prompt = "Continue from where you stopped."
    task["history"].append({"role": "user", "content": resume_prompt})
    task["prompt"] = resume_prompt
    task["status"] = "running"
    task["_stop"] = False
    asyncio.create_task(_run_task(task))
    return {"id": task_id, "status": "running"}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Soft-delete: move task to trash."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_stop"] = True
    provider = _providers.pop(task_id, None)
    if provider:
        await provider.stop()
    task["deleted_at"] = datetime.now().isoformat()
    if task["status"] == "running":
        task["status"] = "stopped"
        task["finished_at"] = datetime.now().isoformat()
    _save(task)
    return {"status": "trashed"}


@app.get("/trash")
async def list_trash():
    """List trashed tasks."""
    result = []
    for t in sorted(_tasks.values(), key=lambda x: x.get("deleted_at") or "", reverse=True):
        if not t.get("deleted_at"):
            continue
        result.append({
            "id": t["id"], "label": t["label"], "status": t["status"],
            "created_at": t["created_at"], "finished_at": t.get("finished_at"),
            "deleted_at": t["deleted_at"],
            "cwd": t["cwd"], "model": t.get("model"),
            "total_cost": t.get("total_cost", 0),
        })
    return result


@app.post("/trash/{task_id}/restore")
async def restore_task(task_id: str):
    """Restore a trashed task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if not task.get("deleted_at"):
        raise HTTPException(400, "Task is not in trash")
    task["deleted_at"] = None
    _save(task)
    return {"status": "restored"}


@app.delete("/trash/{task_id}")
async def permanent_delete(task_id: str):
    """Permanently delete a trashed task."""
    task = _tasks.pop(task_id, None)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_deleted"] = True
    with _db() as con:
        con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    return {"status": "deleted"}


# ── Routes: Git ──────────────────────────────────────────────────────
@app.get("/tasks/{task_id}/gitdiff")
async def get_gitdiff(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return await GitService.get_diff(task["cwd"] or os.getcwd())


@app.get("/git/status")
async def git_status(cwd: str = ""):
    return await GitService.get_status(cwd or os.getcwd())


@app.get("/git/branches")
async def git_branches(cwd: str = ""):
    return await GitService.list_branches(cwd or os.getcwd())


@app.post("/git/branch")
async def git_create_branch(req: GitBranchRequest, cwd: str = ""):
    return await GitService.create_branch(cwd or os.getcwd(), req.name, req.checkout)


@app.post("/git/switch")
async def git_switch_branch(req: GitBranchRequest, cwd: str = ""):
    return await GitService.switch_branch(cwd or os.getcwd(), req.name)


@app.post("/git/commit")
async def git_commit(req: GitCommitRequest, cwd: str = ""):
    return await GitService.commit(cwd or os.getcwd(), req.message, req.add_all)


@app.get("/git/log")
async def git_log(cwd: str = "", n: int = 20):
    return await GitService.get_log(cwd or os.getcwd(), n)


# ── Routes: Workspaces ──────────────────────────────────────────────
@app.get("/workspaces")
async def list_workspaces():
    with _db() as con:
        rows = con.execute("SELECT id, name, path FROM workspaces ORDER BY name").fetchall()
    return [{"id": r[0], "name": r[1], "path": r[2]} for r in rows]


@app.post("/workspaces")
async def add_workspace(req: WorkspaceRequest):
    path = os.path.expanduser(req.path)
    if not os.path.isdir(path):
        raise HTTPException(400, "Path is not a directory")
    name = req.name or os.path.basename(path.rstrip("/"))
    ws_id = str(uuid.uuid4())
    try:
        with _db() as con:
            con.execute("INSERT INTO workspaces (id, name, path) VALUES (?,?,?)", (ws_id, name, path))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Workspace already exists")
    return {"id": ws_id, "name": name, "path": path}


@app.delete("/workspaces/{ws_id}")
async def remove_workspace(ws_id: str):
    with _db() as con:
        n = con.execute("DELETE FROM workspaces WHERE id=?", (ws_id,)).rowcount
    if not n:
        raise HTTPException(404, "Workspace not found")
    return {"status": "deleted"}


@app.get("/analytics")
async def analytics():
    """Return dashboard analytics computed from tasks DB."""
    with _db() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT id, label, status, cwd, mode, model, total_cost,
                   usage, created_at, finished_at, deleted_at
            FROM tasks WHERE deleted_at IS NULL
        """).fetchall()

    tasks_list = [dict(r) for r in rows]
    for t in tasks_list:
        t["usage"] = json.loads(t["usage"] or "{}")
        t["total_cost"] = t["total_cost"] or 0

    total_tasks = len(tasks_list)
    total_cost = sum(t["total_cost"] for t in tasks_list)
    total_input = sum((t["usage"].get("input_tokens", 0) + t["usage"].get("cache_read_input_tokens", 0) + t["usage"].get("cache_creation_input_tokens", 0)) for t in tasks_list)
    total_output = sum(t["usage"].get("output_tokens", 0) for t in tasks_list)
    by_status = {}
    for t in tasks_list:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1

    # Folder breakdown
    folder_map: dict[str, dict] = {}
    for t in tasks_list:
        cwd = t["cwd"] or "unknown"
        # Normalize trailing slashes
        cwd = cwd.rstrip("/")
        if cwd not in folder_map:
            folder_map[cwd] = {"path": cwd, "tasks": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0, "running": 0, "done": 0, "error": 0, "stopped": 0, "models": set()}
        f = folder_map[cwd]
        f["tasks"] += 1
        f["cost"] += t["total_cost"]
        f["input_tokens"] += (t["usage"].get("input_tokens", 0) + t["usage"].get("cache_read_input_tokens", 0) + t["usage"].get("cache_creation_input_tokens", 0))
        f["output_tokens"] += t["usage"].get("output_tokens", 0)
        if t["status"] in f:
            f[t["status"]] += 1
        if t["model"]:
            f["models"].add(t["model"])
    folders = sorted(folder_map.values(), key=lambda x: x["tasks"], reverse=True)
    for f in folders:
        f["models"] = list(f["models"])

    # Agent/model breakdown
    model_map: dict[str, dict] = {}
    for t in tasks_list:
        m = t["model"] or "unknown"
        if m not in model_map:
            model_map[m] = {"model": m, "tasks": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
        e = model_map[m]
        e["tasks"] += 1
        e["cost"] += t["total_cost"]
        e["input_tokens"] += (t["usage"].get("input_tokens", 0) + t["usage"].get("cache_read_input_tokens", 0) + t["usage"].get("cache_creation_input_tokens", 0))
        e["output_tokens"] += t["usage"].get("output_tokens", 0)
    models = sorted(model_map.values(), key=lambda x: x["tasks"], reverse=True)

    # Daily timeline (last 30 days)
    from collections import defaultdict
    daily: dict[str, dict] = defaultdict(lambda: {"date": "", "tasks": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0})
    for t in tasks_list:
        if t["created_at"]:
            day = t["created_at"][:10]
            d = daily[day]
            d["date"] = day
            d["tasks"] += 1
            d["cost"] += t["total_cost"]
            d["input_tokens"] += (t["usage"].get("input_tokens", 0) + t["usage"].get("cache_read_input_tokens", 0) + t["usage"].get("cache_creation_input_tokens", 0))
            d["output_tokens"] += t["usage"].get("output_tokens", 0)
    timeline = sorted(daily.values(), key=lambda x: x["date"])[-30:]

    return {
        "summary": {
            "total_tasks": total_tasks,
            "total_cost": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_status": by_status,
        },
        "folders": folders,
        "models": models,
        "timeline": timeline,
    }


# ── Routes: Git file-level operations ────────────────────────────────
@app.get("/git/changed-files")
async def git_changed_files(cwd: str = ""):
    cwd = cwd or os.getcwd()
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "status", "--porcelain", "-u",
            cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {"error": stderr.decode().strip(), "files": [], "branch": ""}
        files = []
        for line in stdout.decode().splitlines():
            if len(line) < 4:
                continue
            ix, wt = line[0], line[1]
            filepath = line[3:]
            staged = ix not in (" ", "?", "!")
            unstaged = wt not in (" ", "?", "!")
            untracked = ix == "?" and wt == "?"
            files.append({"path": filepath, "index": ix, "working": wt,
                          "staged": staged, "unstaged": unstaged, "untracked": untracked})
        bp = await asyncio.create_subprocess_exec(
            "git", "branch", "--show-current",
            cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        bo, _ = await bp.communicate()
        return {"files": files, "branch": bo.decode().strip()}
    except Exception as e:
        return {"error": str(e), "files": [], "branch": ""}


@app.post("/git/stage")
async def git_stage(req: GitStageRequest, cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "git", "add", "--", *req.files,
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(400, stderr.decode().strip())
    return {"status": "staged", "files": req.files}


@app.post("/git/unstage")
async def git_unstage(req: GitStageRequest, cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "git", "reset", "HEAD", "--", *req.files,
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(400, stderr.decode().strip())
    return {"status": "unstaged", "files": req.files}


@app.get("/git/file-diff")
async def git_file_diff(file: str, cwd: str = "", staged: bool = False):
    cwd = cwd or os.getcwd()
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    cmd.extend(["--", file])
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return {"file": file, "diff": stdout.decode(), "staged": staged}


@app.post("/git/push")
async def git_push(cwd: str = "", set_upstream: bool = False):
    cwd = cwd or os.getcwd()
    cmd = ["git", "push"]
    if set_upstream:
        bp = await asyncio.create_subprocess_exec(
            "git", "branch", "--show-current",
            cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        bo, _ = await bp.communicate()
        branch = bo.decode().strip()
        cmd = ["git", "push", "--set-upstream", "origin", branch]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout.decode() + stderr.decode()).strip()
    if proc.returncode != 0:
        if not set_upstream and ("no upstream" in output.lower() or "set-upstream" in output.lower()):
            return await git_push(cwd=cwd, set_upstream=True)
        raise HTTPException(400, output)
    return {"status": "pushed", "output": output}


@app.get("/git/prs")
async def git_list_prs(cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "gh", "pr", "list",
        "--json", "number,title,author,headRefName,baseRefName,state,url,createdAt,isDraft",
        "--limit", "20",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(400, stderr.decode().strip())
    return json.loads(stdout.decode())


@app.post("/git/pr/create")
async def git_create_pr(req: GitPRCreateRequest, cwd: str = ""):
    cwd = cwd or os.getcwd()
    cmd = ["gh", "pr", "create", "--title", req.title, "--body", req.body, "--base", req.base]
    if req.head:
        cmd.extend(["--head", req.head])
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout.decode() + stderr.decode()).strip()
    if proc.returncode != 0:
        raise HTTPException(400, output)
    return {"status": "created", "url": stdout.decode().strip()}


@app.post("/git/pr/merge")
async def git_merge_pr(req: GitPRMergeRequest, cwd: str = ""):
    cwd = cwd or os.getcwd()
    if req.method not in ("merge", "squash", "rebase"):
        raise HTTPException(400, "method must be merge, squash, or rebase")
    proc = await asyncio.create_subprocess_exec(
        "gh", "pr", "merge", str(req.pr_number), f"--{req.method}",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout.decode() + stderr.decode()).strip()
    if proc.returncode != 0:
        raise HTTPException(400, output)
    return {"status": "merged", "output": output}


@app.get("/git/pr/{pr_number}/diff")
async def git_pr_diff(pr_number: int, cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "gh", "pr", "diff", str(pr_number),
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(400, stderr.decode().strip())
    return {"pr_number": pr_number, "diff": stdout.decode()}


@app.get("/git/all-diffs")
async def git_all_diffs(cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc1 = await asyncio.create_subprocess_exec(
        "git", "diff", "--cached",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    staged_out, _ = await proc1.communicate()
    proc2 = await asyncio.create_subprocess_exec(
        "git", "diff",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    unstaged_out, _ = await proc2.communicate()
    proc3 = await asyncio.create_subprocess_exec(
        "git", "ls-files", "--others", "--exclude-standard",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    untracked_out, _ = await proc3.communicate()
    untracked_diff = ""
    for f in untracked_out.decode().strip().splitlines():
        if f:
            try:
                fpath = os.path.join(cwd, f)
                if os.path.isfile(fpath) and os.path.getsize(fpath) < 50000:
                    with open(fpath, 'r', errors='replace') as fh:
                        content = fh.read()
                    untracked_diff += f"\n--- /dev/null\n+++ b/{f}\n" + "\n".join("+" + line for line in content.splitlines()) + "\n"
            except Exception:
                pass
    return {
        "staged": staged_out.decode(),
        "unstaged": unstaged_out.decode(),
        "untracked": untracked_diff,
        "total": staged_out.decode() + "\n" + unstaged_out.decode() + "\n" + untracked_diff,
    }


async def _generate_commit_message(diff_text: str) -> str:
    """Call vLLM to generate commit message from diff."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "http://192.168.170.76:8000/v1/chat/completions",
                json={
                    "model": "",
                    "messages": [
                        {"role": "system", "content": "Generate a concise conventional commit message (type: description format) for the following diff. Return ONLY the commit message, no explanation. /no_think"},
                        {"role": "user", "content": diff_text[:8000]},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
            )
            data = resp.json()
            msg = data["choices"][0]["message"]["content"].strip()
            msg = msg.strip('`"\'')
            # Strip Qwen3 thinking content
            if '</think>' in msg:
                msg = msg.split('</think>')[-1].strip()
            if '<think>' in msg:
                msg = msg.split('<think>')[0].strip()
            # If msg still has multi-line thinking, take last non-empty line
            lines = [l.strip() for l in msg.splitlines() if l.strip()]
            if len(lines) > 3:
                # Likely thinking leaked — grab lines that look like a commit msg
                for l in reversed(lines):
                    if ':' in l and len(l) < 200:
                        return l.strip('`"\'')
            return lines[-1].strip('`"\'') if lines else msg
    except Exception:
        return f"Update: changes in {diff_text.count('diff --git')} file(s)"


@app.post("/git/ai-commit-message")
async def git_ai_commit_message(cwd: str = ""):
    cwd = cwd or os.getcwd()
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", "--cached",
        cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    diff_text = stdout.decode().strip()
    if not diff_text:
        raise HTTPException(400, "No staged changes to generate commit message from")
    message = await _generate_commit_message(diff_text)
    return {"message": message}


# ── Routes: Scheduler ────────────────────────────────────────────────
@app.get("/scheduler/jobs")
async def list_jobs():
    return scheduler.list_jobs()


@app.post("/scheduler/jobs")
async def create_job(req: CreateJobRequest):
    job = scheduler.add_job(
        name=req.name, prompt=req.prompt, model=req.model,
        mode=req.mode, cwd=req.cwd, schedule=req.schedule,
        interval_seconds=req.interval_seconds,
        delay_seconds=req.delay_seconds, one_shot=req.one_shot,
    )
    return job.to_dict()


@app.delete("/scheduler/jobs/{job_id}")
async def delete_job(job_id: str):
    if scheduler.remove_job(job_id):
        return {"status": "deleted"}
    raise HTTPException(404, "Job not found")


@app.post("/scheduler/jobs/{job_id}/toggle")
async def toggle_job(job_id: str):
    if scheduler.toggle_job(job_id):
        return {"status": "toggled"}
    raise HTTPException(404, "Job not found")


# ── Routes: Monitor ──────────────────────────────────────────────────
@app.get("/monitor")
async def monitor_dashboard():
    return monitor.get_dashboard()


@app.get("/monitor/health")
async def monitor_health():
    return await monitor.check_providers()


@app.get("/monitor/metrics")
async def monitor_metrics():
    return monitor.get_metrics()


# ── Routes: Providers ────────────────────────────────────────────────
@app.get("/providers")
async def providers_list():
    return list_providers()


@app.get("/providers/health")
async def providers_health():
    return await health_check_all()


# ── Routes: Directory autocomplete ───────────────────────────────────
@app.get("/suggest")
async def suggest_path(path: str = ""):
    try:
        path = os.path.expanduser(path) if path else ""
        if not path:
            # Return home subdirs as starting suggestions
            home = os.path.expanduser("~")
            return sorted([
                os.path.join(home, e) for e in os.listdir(home)
                if not e.startswith(".") and os.path.isdir(os.path.join(home, e))
            ])[:20]

        # If path ends with / and is a dir, list its children
        if path.endswith("/") and os.path.isdir(path):
            results = [
                os.path.join(path, e) for e in sorted(os.listdir(path))
                if not e.startswith(".") and os.path.isdir(os.path.join(path, e))
            ]
            return results[:20]

        # If exact dir exists, list children
        if os.path.isdir(path):
            results = [
                os.path.join(path, e) for e in sorted(os.listdir(path))
                if not e.startswith(".") and os.path.isdir(os.path.join(path, e))
            ]
            return results[:20]

        # Partial match: search parent dir for matching names
        search_dir = os.path.dirname(path) or os.path.expanduser("~")
        pattern = os.path.basename(path).lower()

        if os.path.isdir(search_dir):
            quick = sorted([
                os.path.join(search_dir, e) for e in os.listdir(search_dir)
                if not e.startswith(".") and os.path.isdir(os.path.join(search_dir, e))
                and pattern in e.lower()
            ])
            if quick:
                return quick[:20]

        # Deep search across home directory
        home = os.path.expanduser("~")
        proc = await asyncio.create_subprocess_exec(
            "find", home, "-maxdepth", "4", "-type", "d",
            "-iname", f"*{pattern}*", "-not", "-path", "*/.*",
            "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/node_modules/*",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            stdout = b""
        results = [l for l in stdout.decode().splitlines() if "/." not in l and l.strip()]
        # Sort: shorter paths first (more likely what user wants)
        results.sort(key=lambda x: (x.count("/"), len(x)))
        return results[:20]
    except Exception:
        return []


@app.get("/browse")
async def browse_directory(path: str = ""):
    """Browse directory contents for file manager — returns files and dirs."""
    try:
        path = os.path.expanduser(path) if path else os.path.expanduser("~")
        if not os.path.isdir(path):
            return {"error": "Not a directory", "entries": []}
        entries = []
        try:
            items = sorted(os.listdir(path), key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
        except PermissionError:
            return {"path": path, "entries": [], "error": "Permission denied"}
        for name in items:
            if name.startswith('.'):
                continue
            full = os.path.join(path, name)
            is_dir = os.path.isdir(full)
            try:
                stat = os.stat(full)
                size = stat.st_size if not is_dir else None
            except (OSError, PermissionError):
                size = None
            entries.append({
                "name": name,
                "path": full,
                "is_dir": is_dir,
                "size": size,
            })
        return {"path": path, "entries": entries[:200]}
    except Exception as e:
        return {"path": path, "entries": [], "error": str(e)}


# ── Agent definitions (static list for the UI) ──────────────────────
_AGENTS = [
    {"id": "claude-code", "name": "Claude Code", "provider": "claude", "description": "Anthropic's agentic coding assistant"},
    {"id": "opencode",    "name": "OpenCode",    "provider": "opencode", "description": "Open-source agentic coding CLI"},
    {"id": "copilot",     "name": "Copilot",     "provider": "copilot", "description": "GitHub Copilot SDK agent"},
    {"id": "local",       "name": "Local",        "provider": "vllm", "description": "Local vLLM model (Qwen 3.5-9B)"},
]


@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    return _AGENTS


# ── Startup / Shutdown ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await scheduler.start()
    log.info("CC-UI v2 started — %d providers, %d agents, %d tasks loaded, %d scheduled jobs",
             len(list_providers()), len(_AGENTS), len(_tasks), len(scheduler.list_jobs()))


@app.on_event("shutdown")
async def shutdown():
    await scheduler.stop()


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
