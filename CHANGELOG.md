# Changelog

## [2026-04-25] — Session 3: Features, Fixes & Integrations

### Added
- **vLLM Tool Calling**: Full agentic tool loop for Local AI — 8 tools (bash, view, create, edit, grep, glob_search, web_search, web_fetch) with auto-retry up to 25 iterations
- **Tool Manager** (`services/tools/tool_manager.py`): OpenAI-format tool schemas + unified executor leveraging copilot_tools
- **Copilot Tools** (`services/tools/copilot_tools/`): Copied 28 tools from lab for Local AI provider
- **Markdown Rendering**: Full line-by-line parser for agent messages — headings, bold/italic, lists, blockquotes, tables, code blocks, links
- **Trash Tab**: Soft-delete with recovery — backend `deleted_at` column, REST endpoints (`GET /trash`, `POST /trash/{id}/restore`, `DELETE /trash/{id}`), frontend grid with Restore/Delete Forever
- **Agent Filter Sub-tabs**: All / Running / Done filter buttons in nav-right, filters agent cards by status
- **Tool Display Names**: Emoji-prefixed pill labels (⌘ bash, 👁 view, ✏️ edit, etc.) and group headers (🛠 Code Tools, 🌐 Web Tools)
- **Qwen3 Thinking Extraction**: Separates `<think>…</think>` content from visible output, emits as collapsible 💭 Thinking sections

### Fixed
- **CWD Input**: Doubled width (250→500px), dropdown opens **upward** as floating popup, Enter key confirms selection
- **Upload Button**: Cleaner SVG icon, transparent background
- **Filter Tabs Bug**: `updateNavStats()` no longer overwrites `nav-right` innerHTML — updates only `.nav-stats-inner` span
- **Mode Selectors Removed**: Bypass/accept-edits/plan mode UI removed; always bypass

### Changed
- `services/providers/vllm.py`: Complete rewrite with agentic tool-calling loop, thinking extraction, plain-stream fallback
- `backend.py`: Added `deleted_at` column migration, soft-delete logic, trash endpoints, excluded trashed tasks from `GET /tasks`
