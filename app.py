import os
import glob
import json
import httpx
import gradio as gr

BACKEND = "http://localhost:8000"
SESSIONS_DIR = os.path.expanduser("~/.claude/projects")
PERMISSION_MODES = ["bypassPermissions", "acceptEdits", "dontAsk", "default", "plan"]
MODEL_CHOICES = ["claude", "qwen"]

STATUS_ICON = {"running": "⟳", "done": "✓", "stopped": "⏹", "error": "❌"}


def _tasks_samples(tasks: list) -> list:
    """Convert tasks list to Dataset samples: [[display_label], ...]"""
    rows = []
    for t in tasks:
        icon = STATUS_ICON.get(t["status"], "·")
        label = f"{icon} {t['label'][:35]}"
        rows.append([label])
    return rows


# ── Session file helpers ─────────────────────────────────────────────
def _read_sessions(limit=20):
    sessions = []
    for f in glob.glob(f"{SESSIONS_DIR}/**/*.jsonl", recursive=True):
        if "/subagents/" in f:
            continue
        try:
            with open(f) as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            first_user = next((l for l in lines if l.get("type") == "user"), None)
            summary = ""
            if first_user:
                content = first_user.get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                summary = str(content).strip()
                if summary.startswith("<local-command"):
                    summary = ""
            if not summary:
                continue
            sid = os.path.basename(f).replace(".jsonl", "")
            mtime = os.path.getmtime(f)
            sessions.append((mtime, sid, summary[:50]))
        except Exception:
            pass
    sessions.sort(reverse=True)
    return sessions[:limit]


# ── Directory autocomplete ───────────────────────────────────────────
def dir_suggestions(typed: str) -> list[str]:
    typed = os.path.expanduser(typed or "~")
    try:
        base = typed if os.path.isdir(typed) else os.path.dirname(typed)
        prefix = "" if os.path.isdir(typed) else os.path.basename(typed)
        if not os.path.isdir(base):
            return []
        return sorted([
            os.path.join(base, e) for e in os.listdir(base)
            if e.startswith(prefix) and os.path.isdir(os.path.join(base, e))
        ])[:12]
    except Exception:
        return []


# ── Backend calls ────────────────────────────────────────────────────
async def _get_tasks():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND}/tasks", timeout=3)
            return r.json()
    except Exception:
        return []


async def _get_task(task_id: str):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND}/tasks/{task_id}", timeout=3)
            return r.json()
    except Exception:
        return None


async def _create_task(prompt, mode, model, cwd, session_id=None):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{BACKEND}/tasks", json={
                "prompt": prompt, "mode": mode, "model": model,
                "cwd": cwd, "session_id": session_id,
            }, timeout=5)
            return r.json()
    except Exception as e:
        return {"error": str(e)}


async def _send_message(task_id, prompt):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{BACKEND}/tasks/{task_id}/message", json={"prompt": prompt}, timeout=5)
            return r.json()
    except Exception as e:
        return {"error": str(e)}


async def _stop_task(task_id):
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"{BACKEND}/tasks/{task_id}/stop", timeout=3)
    except Exception:
        pass


async def _delete_task(task_id):
    try:
        async with httpx.AsyncClient() as c:
            await c.delete(f"{BACKEND}/tasks/{task_id}", timeout=3)
    except Exception:
        pass


# ── Task list HTML ───────────────────────────────────────────────────
def _tasks_html(tasks: list, active_id: str | None) -> str:
    if not tasks:
        return "<div style='color:#888;padding:8px;font-size:13px'>No tasks yet</div>"
    rows = []
    for t in tasks:
        icon = STATUS_ICON.get(t["status"], "·")
        label = t["label"][:35] + ("…" if len(t["label"]) > 35 else "")
        active = t["id"] == active_id
        bg = "background:#1e3a5f;" if active else ""
        color = "#4fc3f7" if active else "#ccc"
        rows.append(
            f'<div onclick="selectTask(\'{t["id"]}\')" style="cursor:pointer;padding:6px 10px;border-radius:4px;margin:2px 0;{bg}">'
            f'<span style="color:{color};font-size:13px">{icon} {label}</span>'
            f'<div style="color:#666;font-size:11px;margin-top:2px">{t["status"]} · {t["cwd"][-25:]}</div>'
            f'</div>'
        )
    return "".join(rows)


# ── UI handlers ──────────────────────────────────────────────────────
async def init_ui():
    tasks = await _get_tasks()
    return None, [], tasks, "", gr.update(choices=dir_suggestions("~"), value=None)


async def timer_tick(active_id, tasks_cache):
    """Refresh task list + active task history."""
    tasks = await _get_tasks()
    history = []
    if active_id:
        task = await _get_task(active_id)
        if task:
            history = task.get("history", [])
    return tasks, history


async def select_task(task_id, tasks_cache):
    task = await _get_task(task_id)
    if not task:
        return task_id, [], gr.update()
    return task_id, task.get("history", []), task.get("cwd", "")


async def submit_input(prompt, mode, model, cwd, active_id, tasks_cache):
    if not prompt.strip():
        return active_id, [], tasks_cache, ""

    # If active task exists and is done/stopped, send follow-up message
    if active_id:
        task = await _get_task(active_id)
        if task and task["status"] in ("done", "stopped"):
            await _send_message(active_id, prompt)
            return active_id, task.get("history", []) + [{"role": "user", "content": prompt}], tasks_cache, ""

    # Otherwise create new task
    result = await _create_task(prompt, mode, model, cwd)
    if "error" in result:
        return active_id, [{"role": "assistant", "content": f"❌ {result['error']}"}], tasks_cache, ""

    new_id = result["id"]
    tasks = await _get_tasks()
    task = await _get_task(new_id)
    history = task.get("history", []) if task else []
    return new_id, history, tasks, ""


async def stop_active(active_id):
    if active_id:
        await _stop_task(active_id)


async def delete_active(active_id, tasks_cache):
    if active_id:
        await _delete_task(active_id)
    tasks = await _get_tasks()
    new_active = tasks[0]["id"] if tasks else None
    history = []
    if new_active:
        task = await _get_task(new_active)
        history = task.get("history", []) if task else []
    return None, history, tasks


# ── UI ───────────────────────────────────────────────────────────────
with gr.Blocks(title="CC-UI") as demo:

    active_id_state = gr.State(value=None)
    tasks_state = gr.State(value=[])

    with gr.Sidebar(label="⚡ CC-UI", width=240, open=True):
        gr.Markdown("**Directory**")
        cwd_box = gr.Textbox(placeholder="~/path/to/project", show_label=False, container=False,
                             value=os.getcwd())
        dir_dd = gr.Dropdown(choices=[], show_label=False, container=False, allow_custom_value=True)

        gr.Markdown("**Tasks**")
        task_list = gr.HTML(value="<div style='color:#888;font-size:13px'>Loading…</div>", elem_id="task-list")
        with gr.Row():
            del_btn = gr.Button("🗑 delete", size="sm", variant="secondary", scale=1)

        gr.Markdown("**Recent sessions**")
        session_list = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[[s[2]] for s in _read_sessions()],
            label=None, headers=None, samples_per_page=15, type="index",
        )

    chatbot = gr.Chatbot(height="72vh", show_label=False, render_markdown=True,
                         container=False, layout="panel", buttons=[], feedback_options=[])

    msg_box = gr.Textbox(placeholder="New task or follow-up message…", lines=2, max_lines=2,
                         show_label=False, submit_btn=True, container=False)

    with gr.Row():
        model_dd = gr.Dropdown(choices=MODEL_CHOICES, value="claude", show_label=False,
                               scale=0, min_width=100, container=False)
        mode_dd = gr.Dropdown(choices=PERMISSION_MODES, value="bypassPermissions",
                              show_label=False, scale=0, min_width=160, container=False)
        stop_btn = gr.Button("⏹ stop", variant="stop", scale=0)

    timer = gr.Timer(value=2, active=True)

    # JS: clicking task HTML calls this hidden textbox to trigger Python
    task_click_box = gr.Textbox(visible=False, elem_id="task-click-input")

    # Inject JS for task click handling
    demo.load(None, js="""
    () => {
        window.selectTask = function(taskId) {
            const box = document.querySelector('#task-click-input textarea') ||
                        document.querySelector('#task-click-input input');
            if (box) {
                box.value = taskId;
                box.dispatchEvent(new Event('input', {bubbles: true}));
                // trigger change
                const evt = new Event('change', {bubbles: true});
                box.dispatchEvent(evt);
            }
        }
    }
    """)

    # ── Wiring ───────────────────────────────────────────────────────
    demo.load(
        lambda: (None, [], os.getcwd()),
        outputs=[active_id_state, tasks_state, cwd_box],
    )

    # Timer: refresh tasks list + active task history
    async def tick(active_id, tasks_cache):
        tasks = await _get_tasks()
        history = []
        if active_id:
            task = await _get_task(active_id)
            if task:
                history = task.get("history", [])
        html = _tasks_html(tasks, active_id)
        return tasks, history, html

    timer.tick(tick, inputs=[active_id_state, tasks_state], outputs=[tasks_state, chatbot, task_list])

    # Task click via hidden textbox
    task_click_box.change(
        select_task,
        inputs=[task_click_box, tasks_state],
        outputs=[active_id_state, chatbot, cwd_box],
    )

    # Submit: new task or follow-up
    msg_box.submit(
        submit_input,
        inputs=[msg_box, mode_dd, model_dd, cwd_box, active_id_state, tasks_state],
        outputs=[active_id_state, chatbot, tasks_state, msg_box],
    )

    stop_btn.click(stop_active, inputs=[active_id_state])

    del_btn.click(
        delete_active,
        inputs=[active_id_state, tasks_state],
        outputs=[active_id_state, chatbot, tasks_state],
    ).then(
        lambda tasks, active_id: _tasks_html(tasks, active_id),
        inputs=[tasks_state, active_id_state],
        outputs=[task_list],
    )

    # Dir autocomplete
    cwd_box.change(lambda t: gr.update(choices=dir_suggestions(t)), inputs=[cwd_box], outputs=[dir_dd])
    dir_dd.select(lambda v: v, inputs=[dir_dd], outputs=[cwd_box])

    # Load session into new task
    def load_session_history(sid):
        history = []
        for f in glob.glob(f"{SESSIONS_DIR}/**/{sid}.jsonl", recursive=True):
            try:
                with open(f) as fh:
                    for line in fh:
                        if not line.strip(): continue
                        obj = json.loads(line)
                        if obj.get("type") == "user":
                            content = obj.get("message", {}).get("content", "")
                            if isinstance(content, list):
                                content = " ".join(c.get("text","") for c in content if isinstance(c,dict))
                            content = str(content).strip()
                            if content and not content.startswith("<local-command"):
                                history.append({"role": "user", "content": content})
                        elif obj.get("type") == "assistant":
                            for block in obj.get("message", {}).get("content", []):
                                if isinstance(block, dict) and block.get("type") == "text":
                                    history.append({"role": "assistant", "content": block["text"]})
            except Exception:
                pass
            break
        return history

    async def on_session_select(evt: gr.SelectData):
        sessions = _read_sessions()
        if evt.index >= len(sessions):
            return gr.update(), gr.update()
        sid = sessions[evt.index][1]
        # Create a "viewer" task with this session loaded
        history = load_session_history(sid)
        return history, sid

    session_list.select(on_session_select, outputs=[chatbot, active_id_state])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean(), footer_links=[])
