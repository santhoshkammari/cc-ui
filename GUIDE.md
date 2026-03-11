# CC-UI Build Guide

## What this is
A Gradio UI for Claude Code — terminal power, visual interface.

## Core principles
- No fancy UI. No verbose code. Clean, minimal, readable.
- Separate sections neatly using Gradio sidebar layout.
- Leverage existing Gradio components — don't reinvent.
- Start basic, add features incrementally.

## Before building any feature
1. Check if Gradio is up to date: `uv pip install --upgrade gradio`
2. Read `CLAUDE_CODE_AGENT_SDK_PYTHON_DOCUMENTATION.md` in this repo for Claude Code/Agent SDK reference — use it before guessing APIs.
3. Read the installed Gradio package source to understand available components:
   - `python -c "import gradio; print(gradio.__file__)"` → get package path
   - Browse `gradio/components/` to see what's available
4. Pick the right component first — saves time over custom workarounds.

## Stack
- Python + Gradio (latest)
- `uv` for package management (not pip)
- No extra frameworks unless absolutely needed

## Layout convention
- Use Gradio sidebar for navigation between sections
- Each major feature = its own section/tab
- Keep each section self-contained

## Build order (step by step)
1. Basic shell — sidebar + placeholder sections
2. Run claude code commands, see output
3. Session/history view
4. File browser
5. ... (features added incrementally)

## Code style
- Short functions, no over-engineering
- No docstrings on obvious functions
- Delete unused code, don't comment it out

## Tracking findings
- Always maintain `FINDINGS.md` — log SDK gotchas, Gradio API quirks, failures, and solutions
- Read `FINDINGS.md` at the start of each session before writing any code — avoids repeating mistakes
- Update it whenever something surprising or tricky is discovered
