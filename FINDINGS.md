# Findings / Failures / Learnings

## CRITICAL

- **Must `os.environ.pop("CLAUDECODE", None)` at app startup** — running inside Claude Code sets this env var, blocking nested subprocess launch
- **Use top-level `query()` NOT `ClaudeSDKClient`** — `query()` is simpler async iterator; for multi-turn chat use `resume=session_id` (get `session_id` from `ResultMessage`)
- **`dangerouslySkipPermissions` is NOT valid** — CLI only accepts: `bypassPermissions`, `acceptEdits`, `dontAsk`, `default`, `plan`
- **Tool results come as `UserMessage`, NOT `ToolResultBlock`** — pattern is `AssistantMessage(ToolUseBlock) → UserMessage → AssistantMessage(ToolUseBlock) → UserMessage → AssistantMessage(TextBlock)`. Each tool is its own separate `AssistantMessage`.
- **To flip tool accordion pending→done**: do it when the NEXT `AssistantMessage` arrives (= previous tool completed), not when `ToolResultBlock` is seen
- **Do NOT `break` on `ResultMessage`** — causes `RuntimeError: Attempted to exit cancel scope in a different task`. Let the iterator exhaust naturally.

## SDK

- `query()` async iterator is the right approach — `ClaudeSDKClient` is more complex with no benefit for simple chat
- Multi-turn: save `session_id` from `ResultMessage`, pass as `resume=session_id` in next `ClaudeAgentOptions`
- `SystemMessage` subtypes `"init"` and `"system"` are noisy — skip them

## Gradio

- Gradio 6.x: `gr.Chatbot` does NOT have a `type` parameter — natively uses `MessageDict` `{"role": ..., "content": ...}`
- Gradio 6.x: `theme` and `css` params moved from `gr.Blocks(...)` to `demo.launch(...)`
- **Don't use custom CSS for theming** — use `gr.themes.Ocean()` in `launch(theme=...)`
- `return value` inside async generator (has `yield`) is SyntaxError — use `yield value; return`
- `gr.Dataset` uses `samples=` not `choices=`; update via `gr.update(samples=...)`
- `gr.Chatbot` collapsible tool accordions: use `metadata={"title": "...", "status": "pending"|"done"}` on messages. `"pending"` = open+spinner, `"done"` = closed
- `gr.Chatbot(buttons=[])` removes all copy/share buttons; `feedback_options=[]` removes like/dislike
- `demo.launch(footer_links=[])` hides the "Use via API · Built with Gradio · Settings" footer
- `gr.Chatbot(layout="panel")` removes bubble containers around messages — flat clean look
- `gr.Chatbot(container=False)` removes outer border of the chatbot component itself
- `show_api=False` is NOT a valid `launch()` param in Gradio 6 — use `footer_links=[]` instead
