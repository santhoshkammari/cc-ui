# CC-OS Agents UI — Architecture

## Overview

CC-OS (Ship · Sleep · Repeat) is a multi-agent orchestration UI built as a single-page application (`index.html`) backed by a FastAPI server (`backend.py`). It manages AI coding agents (Claude Code, OpenCode, Copilot, Local/vLLM) through a unified interface.

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│                   Browser (SPA)                  │
│  index.html — HTML + CSS + Vanilla JS            │
│                                                  │
│  ┌────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Nav Bar │  │Agent Grid│  │ Sidebar / Detail │  │
│  └────────┘  └──────────┘  └──────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │              Chat Bar (input)              │  │
│  └────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────┘
               │ REST API (fetch) + Polling (2.5s)
               ▼
┌──────────────────────────────────────────────────┐
│              FastAPI Backend (backend.py)         │
│                                                  │
│  Routes:                                         │
│  ├── GET/POST /tasks        (CRUD agents)        │
│  ├── POST /tasks/:id/message (follow-up)         │
│  ├── POST /tasks/:id/stop   (stop agent)         │
│  ├── POST /tasks/:id/resume (resume agent)       │
│  ├── DELETE /tasks/:id      (delete agent)        │
│  ├── GET /git/*             (git operations)     │
│  ├── GET/POST /workspaces   (workspace mgmt)     │
│  └── GET /suggest           (CWD autocomplete)   │
│                                                  │
│  Services:                                       │
│  ├── Provider Registry (claude, opencode, etc.)  │
│  ├── Scheduler (cron/delayed/recurring)          │
│  ├── Monitor (health, metrics)                   │
│  └── GitService (branches, commits, diffs)       │
│                                                  │
│  Storage: SQLite (tasks.db)                      │
│  In-memory: _tasks dict + _providers dict        │
└──────────────────────────────────────────────────┘
```

---

## Frontend Components

### 1. App Shell (`#app`)
- Flex column layout, full viewport height
- Contains: Nav Bar → Tab Views → Detail View

### 2. Top Navigation (`#top-nav`)
- Fixed 48px height bar
- Brand logo + tabs (Agents, Workspaces, Settings)
- Right-side stats: running count, total agents, cost

### 3. Agents View (`#view-agents`)
- **Default active tab**
- Contains:
  - **Agent Grid** (`#agent-grid`): 2-column CSS grid of agent cards
  - **Sidebar** (`#agent-sidebar`): Flex-based right panel, opens on single-click
  - **Chat Bar** (`#chat-bar`): Bottom input for launching new agents

### 4. Agent Cards (`.agent-card`)
- **2-line compact layout** with dynamic height (min-height 58px)
- **Line 1** (`.ac-line1`): Status dot + ticker text (label + activity preview)
- **Line 2** (`.ac-line2`): Metadata badges + action buttons
  - Metadata: directory, git branch, agent type badge, tokens, start/end timestamps
  - Actions (hover): stop, resume, delete, diff buttons
- **Status indicators**: Left border color (orange=running, green=done, red=error, gray=stopped)
- **Click behavior**:
  - Single-click → Opens sidebar (conversation preview)
  - Double-click → Opens full detail view (conversation + git)

### 5. Agent Sidebar (`#agent-sidebar`)
- **Flex-based panel** (not overlay) — grid reflows when sidebar opens
- Width: 30% (CSS variable `--sidebar-w`)
- Shows: Header with agent label + close button, scrollable conversation
- Responsive: 50% on tablets, 100% on mobile

### 6. Detail View (`#detail-view`)
- **Full-screen replacement** of agents view (hides `#view-agents`)
- Split layout:
  - **Left** (70%): Conversation history + follow-up input
  - **Right** (30%): Git changes panel (branch, staged/unstaged files, diffs, commit/push)
- Back button returns to agents grid

### 7. Chat Bar (`#chat-bar`)
- Pill-shaped input with auto-resize textarea
- Controls: File attach, agent selector, mode selector, CWD autocomplete
- Drag-and-drop file attachment support
- Keyboard: Enter to send, Shift+Enter for newline

### 8. Workspaces View (`#view-workspace`)
- Grid of workspace cards showing path and agent count
- Add/remove workspace dialog

### 9. Settings View (`#view-profile`)
- Theme grid (15+ themes: GitHub Dark/Light, Nord, Catppuccin, etc.)
- vLLM configuration (IP, port, model)
- Glass theme support with backdrop-filter

---

## Data Flow

### Agent Lifecycle
```
User types prompt → POST /tasks → Backend creates task
  → Provider.run() streams events → History updated in-memory
  → Frontend polls GET /tasks every 2.5s → Cards re-render
  → Task completes → status="done", finished_at set
```

### Event Types from Providers
- `TEXT` — Streamed assistant text
- `TOOL_START` / `TOOL_RESULT` — Tool call lifecycle
- `THINKING` — Model thinking/reasoning
- `AGENT_GROUP` — Sub-agent delegation
- `COST` — Token usage and cost tracking
- `ERROR` / `DONE` — Terminal events

### Conversation History Format
```javascript
[
  { role: "user", content: "..." },
  { role: "assistant", content: "...", metadata: { title: "⚙ tool", status: "done" } },
  { role: "agent-group", agent_id: "...", status: "running", children: [...] }
]
```

### Tool Call Grouping

Consecutive tool calls from the same category are grouped into compact horizontal pill rows:

```
Categories:
  code  → bash, read, edit, write, glob, grep, list, find
  web   → websearch, web_search, fetch
  other → each tool gets its own category
```

**Visual structure:**
```
▶  10  code                       ← collapsed group header (click to expand)

▼  10  code                       ← expanded group
  [✓ glob] [❌ read] [✓ bash] [✓ bash] [✓ bash]  ← horizontal pills
  [✓ bash] [✓ bash] [✓ bash] [✓ bash] [✓ bash]
  ┌─ INPUT ──────────────────┐    ← active pill content
  │  {"pattern": "*"}        │
  ├─ OUTPUT ─────────────────┤
  │  /home/.../file1         │
  └──────────────────────────┘
```

**State preservation**: Expanded groups and active pills persist across 2.5s poll re-renders via `_tgState` map (sidebar/detail tracked independently). State resets when sidebar/detail closes.

**Mode**: Always `bypassPermissions` (mode selector removed from UI).

---

## Layout System

### Grid Reflow on Sidebar Open
```
┌─────────────────────────────────┐     ┌──────────────┬──────────┐
│  Card  │  Card  │               │     │  Card │ Card │ Sidebar  │
│  Card  │  Card  │               │ ──► │  Card │ Card │ (conv)   │
│  Card  │        │               │     │  Card │      │          │
│        │        │               │     │       │      │          │
└─────────────────────────────────┘     └──────────────┴──────────┘
      Grid (flex: 1)                    Grid (flex:1)   30% width
```

The sidebar uses **flex-based layout** (not absolute positioning) so the grid naturally shrinks when the sidebar opens, preventing overlap.

### Detail View Takeover
```
Agents View (hidden)        Detail View (visible)
                           ┌───────────┬─────────┐
                           │  Back btn  │ cost/tk │
                           ├───────────┤ Git     │
                           │           │ Changes │
                           │  Convo    │ Files   │
                           │  History  │ Diff    │
                           │           │ Commit  │
                           ├───────────┤         │
                           │ Follow-up │         │
                           └───────────┴─────────┘
```

---

## Backend Architecture

### Task Storage
- **In-memory**: `_tasks` dict for fast access during streaming
- **SQLite** (`tasks.db`): Persistent storage, synced on every event
- On startup: loads all tasks from DB, marks interrupted "running" tasks as "stopped"

### Provider System
- Registry pattern: `get_provider(name)` returns provider instance
- Each provider implements async `run(prompt, config, history)` generator
- Supported: Claude Code, OpenCode, Copilot, Local/vLLM

### Agent-to-Provider Mapping
```python
{
  "claude-code": "claude",
  "opencode":    "opencode",
  "copilot":     "copilot",
  "local":       "vllm",
}
```

### Advisor System (Optional)
- After worker completes, an advisor provider can review the output
- Adds an "agent-group" entry to history with review feedback
- Max 2 review rounds

---

## Theming

15 built-in themes using CSS custom properties:
- GitHub Dark/Light, One Dark, Nord, Catppuccin, Gruvbox, Monokai, Solarized
- macOS Dark/Light/Glass, Claude Dark/Light, Cursor Dark

Glass themes add `backdrop-filter: blur()` for frosted glass effect.

---

## Key Files

| File | Purpose |
|------|---------|
| `index.html` | Complete SPA (HTML + CSS + JS) |
| `backend.py` | FastAPI server, task management, provider orchestration |
| `services/providers/` | Provider implementations (claude, opencode, etc.) |
| `services/scheduler.py` | Cron/delayed/recurring job scheduling |
| `services/monitor.py` | Health and metrics monitoring |
| `services/git_service.py` | Git operations (branch, commit, diff, push) |
| `tasks.db` | SQLite database for persistent task storage |
