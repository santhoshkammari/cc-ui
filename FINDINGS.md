# Findings / Failures / Learnings

## CRITICAL

## CRITICAL

- **Must `os.environ.pop("CLAUDECODE", None)` at app startup** — running inside Claude Code sets this env var, blocking nested subprocess launch
- **Use top-level `query()` NOT `ClaudeSDKClient`** — `query()` is simpler async iterator; for multi-turn chat use `resume=session_id` (get `session_id` from `ResultMessage`)
- **`dangerouslySkipPermissions` is NOT valid** — CLI only accepts: `bypassPermissions`, `acceptEdits`, `dontAsk`, `default`, `plan`

## SDK

- `client.query()` + `client.receive_response()` is the right pattern for multi-turn chat (vs one-off `query()`)
- Gradio 6.x: `css` param moved from `gr.Blocks(css=...)` to `demo.launch(css=...)` — same as `theme`
- **Don't use custom CSS for theming** — use `gr.themes.Ocean()` passed to `launch(theme=...)`. Avoids layout bugs and mobile-centering from `max-width` CSS

## Gradio

- Gradio 6.x: `gr.Chatbot` does NOT have a `type` parameter — it natively uses `MessageDict` format `{"role": ..., "content": ...}`
- Gradio 6.x: `theme` param moved from `gr.Blocks(theme=...)` to `demo.launch(theme=...)` — use launch() instead
- For streaming responses, yield intermediate states from the generator; Gradio updates UI on each yield
- `msg_box.submit(...)` with `submit_btn=True` on Textbox gives Enter-to-send + button in one widget
- `return value` inside an async generator (function with `yield`) is a SyntaxError — use `yield value; return` instead
- Always `uv pip install --upgrade gradio` before building — API changes between versions

## Message Types (SDK)

- `AssistantMessage.content` = list of `TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock`
- `dangerouslySkipPermissions` is NOT a valid CLI permission mode — use `bypassPermissions` instead. Valid modes: `bypassPermissions`, `acceptEdits`, `dontAsk`, `default`, `plan`
- `ToolResultBlock`: `.content` (str or list of dicts), `.is_error` — result of tool execution
- `ThinkingBlock`: `.thinking` (str) — extended thinking text
- `SystemMessage`: `.subtype`, `.data` — task_started/task_progress/task_notification etc
- `ResultMessage`: `.is_error`, `.total_cost_usd`, `.usage`, `.session_id` — final sentinel
- `dangerouslySkipPermissions` is accepted at runtime even though not in typed `PermissionMode` Literal
