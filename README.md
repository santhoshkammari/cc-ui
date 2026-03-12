# cc-ui

A minimal multi-agent task manager UI for Claude Code CLI.

Run multiple Claude Code agents in parallel. Spawn tasks, switch between them, watch them live, stop or delete them.

## Stack

- **Backend**: FastAPI — manages tasks in-memory, runs `claude_agent_sdk` async in background
- **Frontend**: Plain HTML/JS — no framework, talks directly to the backend API

## Setup

```bash
pip install -e .
```

## Usage

```bash
ccui start   # starts backend at http://127.0.0.1:8000
ccui stop    # stops it
```

Open **http://127.0.0.1:8000** in your browser.

## Features

- **+ New** button to start a fresh task
- Tasks run in background — switch between them freely
- Live polling every 2s — watch tools running in real time
- Stop or delete any task from the sidebar
- Follow-up messages continue the same Claude session
- Set working directory per task
- Supports Claude and Qwen models
- Permission mode selector (bypassPermissions, acceptEdits, etc.)
