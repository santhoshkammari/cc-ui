"""
CC-UI v2 Backend — Unified FastAPI gateway.

Integrates all services:
- Provider registry (10+ AI backends, swappable)
- Task management with unified provider interface
- Scheduler (cron/delayed/recurring jobs)
- Monitor (health, metrics, system info)
- Git service (branches, commits, diffs)
- Directory autocomplete

Original backend preserved as backend_legacy.py for reference.
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
        for col, ctype in [("finished_at","TEXT"),("total_cost","REAL"),("usage","TEXT"),("branch","TEXT")]:
            try:
                con.execute(f"ALTER TABLE tasks ADD COLUMN {col} {ctype}")
                con.commit()
            except Exception:
                pass

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
    with _db() as con:
        con.execute("""INSERT OR REPLACE INTO tasks
            (id, label, status, history, session_id, cwd, mode, model, prompt, created_at, finished_at, total_cost, usage, branch)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task["id"], task["label"], task["status"],
             json.dumps(task["history"]), task["session_id"],
             task["cwd"], task["mode"], task["model"],
             task["prompt"], task["created_at"], task.get("finished_at"),
             task.get("total_cost", 0), json.dumps(task.get("usage", {})),
             task.get("branch", "")))

def _load_all() -> dict[str, dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    result = {}
    cols = ["id","label","status","history","session_id","cwd","mode","model","prompt","created_at","finished_at","total_cost","usage","branch"]
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
    try:
        provider = get_provider(model_name)
    except ValueError as e:
        task["history"].append({"role": "assistant", "content": f"❌ {e}"})
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        return

    _providers[task["id"]] = provider

    config = ProviderConfig(
        model=task.get("extra", {}).get("vllm_model", ""),
        mode=task["mode"],
        cwd=task["cwd"],
        session_id=task["session_id"],
        base_url=task.get("extra", {}).get("vllm_url", ""),
        api_key=task.get("extra", {}).get("vllm_key", ""),
        extra=task.get("extra", {}),
    )

    history_snapshot = task["history"][:]
    tool_calls: list[list] = []  # [title, content, status]
    text_buf = ""

    def snapshot():
        tools = [{"role": "assistant", "content": c, "metadata": {"title": t, "status": s}} for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text_buf}] if text_buf else []
        return history_snapshot + tools + cur

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
                # Add/update agent group in history
                agent_id = event.metadata.get("agent_id", "")
                found = False
                for entry in task["history"]:
                    if entry.get("role") == "agent-group" and entry.get("agent_id") == agent_id:
                        entry.update(event.metadata)
                        entry["role"] = "agent-group"
                        found = True
                        break
                if not found:
                    entry = {"role": "agent-group"}
                    entry.update(event.metadata)
                    task["history"].append(entry)

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
        task["history"] = snapshot() + [{"role": "assistant", "content": f"❌ {e}"}]
        task["status"] = "error"
        task["finished_at"] = datetime.now().isoformat()
        _save(task)
        _providers.pop(task["id"], None)
        return

    for tc in tool_calls:
        tc[2] = "done"
    final = [{"role": "assistant", "content": text_buf}] if text_buf else []
    if not tool_calls and not text_buf:
        final = [{"role": "assistant", "content": "*(no response)*"}]
    task["history"] = history_snapshot + [
        {"role": "assistant", "content": c, "metadata": {"title": t, "status": s}}
        for t, c, s in tool_calls
    ] + final
    task["status"] = "done"
    task["finished_at"] = datetime.now().isoformat()
    _save(task)
    _providers.pop(task["id"], None)


# ── API Models ───────────────────────────────────────────────────────
class CreateTaskRequest(BaseModel):
    prompt: str
    mode: str = "bypassPermissions"
    model: str = "claude"
    cwd: str = ""
    session_id: str | None = None
    branch: str = ""
    extra: dict = {}

class SendMessageRequest(BaseModel):
    prompt: str

class CreateJobRequest(BaseModel):
    name: str
    prompt: str
    model: str = "claude"
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


# ── Routes: Static ───────────────────────────────────────────────────
@app.get("/")
async def index():
    static_index = os.path.join(HERE, "static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
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


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    task = _tasks.pop(task_id, None)
    if not task:
        raise HTTPException(404, "Task not found")
    task["_stop"] = True
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
            return []

        if os.path.isdir(path):
            results = [
                os.path.join(path, e) for e in sorted(os.listdir(path))
                if not e.startswith(".") and os.path.isdir(os.path.join(path, e))
            ]
            return results[:15]

        search_dir = os.path.dirname(path) or os.path.expanduser("~")
        pattern = os.path.basename(path).lower()

        if os.path.isdir(search_dir):
            quick = [
                os.path.join(search_dir, e) for e in sorted(os.listdir(search_dir))
                if not e.startswith(".") and os.path.isdir(os.path.join(search_dir, e))
                and pattern in e.lower()
            ]
            if quick:
                return quick[:15]

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


# ── Startup / Shutdown ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await scheduler.start()
    log.info("CC-UI v2 started — %d providers, %d tasks loaded, %d scheduled jobs",
             len(list_providers()), len(_tasks), len(scheduler.list_jobs()))


@app.on_event("shutdown")
async def shutdown():
    await scheduler.stop()


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
