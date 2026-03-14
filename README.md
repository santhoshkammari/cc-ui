# CC-UI

A web UI for running Claude Code (and other AI agents) as parallel background tasks, with a Kanban dashboard and live git diff panel.

## Stack

- **Backend**: FastAPI + SQLite (`backend.py`) — API server + task runner
- **Frontend**: Single-file plain HTML/CSS/JS (`index.html`) — no build step, no framework

## Running

```bash
python backend.py        # starts on http://0.0.0.0:8001
```

---

## Architecture: Two Layers

### Layer 1 — Kanban Dashboard (default view)

The home screen. Shows all tasks as cards in three columns:

| Column | Statuses shown |
|--------|----------------|
| **Running** | `running` |
| **Done** | `done` |
| **Stopped / Error** | `stopped`, `error` |

**Creating a task:** Type in the bottom chat bar, optionally set a working directory (with autocomplete), pick model/mode, press Enter. Task card appears in Running column. **You stay on Layer 1.**

**Opening a task:** Click any card → opens Layer 2.

**Left nav sidebar:** Decorative — Tasks (active), Projects, Code, Notes, Settings are placeholders for future features.

---

### Layer 2 — Task Detail (chat view)

Opens when you click a task card. Split into two panels:

#### Left panel (75%) — Chat
- Full conversation history: user messages, assistant text, tool call accordions (click to expand)
- Follow-up textarea at the bottom — sends to the same Claude session
- `← Back` returns to Layer 1

#### Right panel (25%) — Git Diff
- Auto-detects if the task's `cwd` is a git repo
- **STATUS** section: `git status --short`
- **DIFF** section: `git diff HEAD` with syntax highlighting
  - Green = additions, Red = deletions, Blue = file headers
- Refreshes every 2s while running; shows "Not a git repo" gracefully

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` |
| `GET` | `/tasks` | List all tasks (summary) |
| `POST` | `/tasks` | Create + start a task |
| `GET` | `/tasks/{id}` | Full task with history |
| `POST` | `/tasks/{id}/message` | Follow-up to existing session |
| `POST` | `/tasks/{id}/stop` | Stop running task |
| `DELETE` | `/tasks/{id}` | Delete task |
| `GET` | `/tasks/{id}/gitdiff` | Git status + diff for task's cwd |
| `GET` | `/suggest?path=` | Directory autocomplete |

---

## Task Object

```python
{
  "id": str,           # UUID
  "label": str,        # first 40 chars of prompt
  "status": str,       # running | done | stopped | error
  "history": list,     # messages + tool calls (see below)
  "session_id": str,   # Claude session ID (enables follow-up)
  "cwd": str,          # working directory
  "mode": str,         # bypassPermissions | acceptEdits | dontAsk | default | plan
  "model": str,        # claude | qwen | opencode
  "prompt": str,       # latest prompt sent
  "created_at": str,   # ISO timestamp
  "_stop": bool,       # in-memory signal to abort runner
}
```

### History message formats
```python
{"role": "user", "content": "..."}                                         # user prompt
{"role": "assistant", "content": "..."}                                    # text response
{"role": "assistant", "content": "...", "metadata": {"title": "...", "status": "done|pending"}}  # tool call
```

---

## Supported Models

| Value | Runner |
|-------|--------|
| `claude` | `claude_agent_sdk.query()` async stream |
| `qwen` | `qwen --output-format stream-json` subprocess |
| `opencode` | `opencode run --format json` subprocess |

---

## Persistence

Tasks persist in `tasks.db` (SQLite). History stored as JSON. On startup, tasks in `running` state are marked `stopped` (process died).

---

## Frontend JS Key Functions

| Function | Purpose |
|----------|---------|
| `showLayer(1\|2)` | Toggle between Layer 1 (dashboard) and Layer 2 (chat) |
| `openTask(id)` | Switch to Layer 2, load task chat + git diff |
| `renderKanban()` | Re-render all three Kanban columns from `tasks[]` |
| `sendNewTask()` | Read Layer 1 bar → POST /tasks → stay on Layer 1 |
| `sendPrompt()` | Send follow-up in Layer 2 |
| `refreshGitPanel()` | Fetch /gitdiff and render right sidebar |
| `renderDiff(text)` | Parse raw git diff text → colored HTML |
| `pollActive()` | Every 2s: refresh tasks list + chat history + git panel |
| `timeAgo(iso)` | Convert ISO timestamp to "Xm ago" string |
