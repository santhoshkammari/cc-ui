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

### POST /tasks request body

```python
{
  "prompt": str,           # Required. The initial prompt/command
  "mode": str,            # Permission mode: bypassPermissions | acceptEdits | dontAsk | default | plan (default: bypassPermissions)
  "model": str,           # Model to use: claude | qwen | opencode | vllm (default: claude)
  "cwd": str,             # Working directory for task execution (default: current dir)
  "session_id": str,      # Optional. Resume existing session
  "vllm_url": str,        # Optional. vLLM server URL (default: http://192.168.170.76:8000)
  "vllm_key": str,        # Optional. vLLM API key (default: "dummy")
  "vllm_model": str,      # Optional. Model path on vLLM server
}
```

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
| `vllm` | `claude_agent_sdk.query()` via vLLM server (local inference) |

---

## Backend Architecture

### Task Runners

Each model has a dedicated async runner that processes tasks:

- **`_run_claude()`** — Uses `claude_agent_sdk.query()` to stream responses from Claude API
- **`_run_vllm()`** — Routes to local vLLM server via Claude SDK with custom `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`
- **`_run_qwen()`** — Spawns `qwen` subprocess, parses streaming JSON output
- **`_run_opencode()`** — Spawns `opencode run` subprocess, parses JSON events

Each runner:
1. Takes a task snapshot to preserve history up to that point
2. Streams responses block-by-block (text, thinking, tool calls, tool results)
3. Updates `task["history"]` in real-time
4. On completion: marks status as `done`, stores `finished_at` timestamp, persists to DB
5. On error/stop: marks as `error`/`stopped`, records final message

The `task["_stop"]` flag allows graceful cancellation: runners check it on each message and exit early if set.

---

## Persistence

Tasks persist in `tasks.db` (SQLite). History stored as JSON. On startup, tasks in `running` state are marked `stopped` (process died).

---

## Frontend Structure

The entire frontend is a **single 54KB HTML file** (`index.html`) with embedded CSS and JavaScript—no build step, no framework, vanilla DOM manipulation.

### Key Design

- **Global state**: `tasks[]` array holds all tasks from server
- **Two layers**: controlled by `currentLayer` variable
- **Real-time polling**: `pollActive()` runs every 2s while any task is running
- **History rendering**: Messages + tool calls with collapsible accordions for tool details
- **Git diff**: Parsed and colorized with regex patterns (green/red/blue)

### Key Functions

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

### UI Sections

- **Left sidebar**: Task list (active), Projects, Code, Notes, Settings (placeholders)
- **Layer 1 main**: Kanban board with Running/Done/Stopped columns, bottom chat bar
- **Layer 2 left (75%)**: Chat history with message/tool call rendering, follow-up textarea
- **Layer 2 right (25%)**: Git status + diff panel (auto-refreshes, colorized)
