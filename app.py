import os
import json
import glob
import gradio as gr
from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    AssistantMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,
)

os.environ.pop("CLAUDECODE", None)

PERMISSION_MODES = ["bypassPermissions", "acceptEdits", "dontAsk", "default", "plan"]
SKIP_SUBTYPES = {"init", "system"}
SESSIONS_DIR = os.path.expanduser("~/.claude/projects")

session_id: str | None = None
_stop = False


# ── Session helpers ────────────────────────────────────────────────
def _read_sessions(limit=50):
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
            sessions.append((mtime, sid, summary[:55]))
        except Exception:
            pass
    sessions.sort(reverse=True)
    return sessions[:limit]


def get_session_samples():
    return [[s[2]] for s in _read_sessions()]


def load_session_history(sid: str) -> list:
    for f in glob.glob(f"{SESSIONS_DIR}/**/{sid}.jsonl", recursive=True):
        try:
            history = []
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("type") == "user":
                        content = obj.get("message", {}).get("content", "")
                        if isinstance(content, list):
                            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                        content = str(content).strip()
                        if content and not content.startswith("<local-command"):
                            history.append({"role": "user", "content": content})
                    elif obj.get("type") == "assistant":
                        for block in obj.get("message", {}).get("content", []):
                            if isinstance(block, dict) and block.get("type") == "text":
                                history.append({"role": "assistant", "content": block["text"]})
            return history
        except Exception:
            pass
    return []


# ── Handlers ───────────────────────────────────────────────────────
def make_options(mode: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(permission_mode=mode, resume=session_id)


def tool_msg(title: str, content: str, status: str = "done") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "metadata": {"title": title, "status": status},
    }



async def send_message(prompt: str, mode: str, history: list):
    global session_id, _stop
    if not prompt.strip():
        yield history, ""
        return

    _stop = False
    history = history + [{"role": "user", "content": prompt}]
    yield history, ""

    # Each entry: [title, content, status]
    tool_calls: list[list] = []
    text = ""

    def snapshot():
        tools = [tool_msg(t, c, s) for t, c, s in tool_calls]
        cur = [{"role": "assistant", "content": text}] if text else []
        return history + tools + cur

    def close_last_pending():
        # flip the most recent pending tool to done
        for tc in reversed(tool_calls):
            if tc[2] == "pending":
                tc[2] = "done"
                break

    async for msg in query(prompt=prompt, options=make_options(mode)):
        if _stop:
            for tc in tool_calls:
                tc[2] = "done"
            yield snapshot() + [{"role": "assistant", "content": "⏹ *stopped*"}], ""
            return

        if isinstance(msg, AssistantMessage):
            if msg.error:
                yield history + [{"role": "assistant", "content": f"❌ {msg.error}"}], ""
                return
            # New AssistantMessage = previous tool is done
            close_last_pending()
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text += block.text
                elif isinstance(block, ThinkingBlock):
                    tool_calls.append(["💭 Thinking", block.thinking[:500], "pending"])
                elif isinstance(block, ToolUseBlock):
                    args = json.dumps(block.input, indent=2) if block.input else ""
                    tool_calls.append([f"⚙ {block.name}", f"```json\n{args}\n```", "pending"])
                elif isinstance(block, ToolResultBlock):
                    content = block.content or ""
                    if isinstance(content, list):
                        content = "\n".join(c.get("text", str(c)) for c in content)
                    preview = str(content)[:600] + ("…" if len(str(content)) > 600 else "")
                    close_last_pending()
                    tool_calls.append([f"{'❌' if block.is_error else '✓'} result", f"```\n{preview}\n```", "done"])
            yield snapshot(), ""

        elif isinstance(msg, SystemMessage):
            if msg.subtype not in SKIP_SUBTYPES:
                desc = getattr(msg, "description", None) or msg.subtype
                tool_calls.append([f"⚙ {desc}", "", "done"])
                yield snapshot(), ""

        elif isinstance(msg, ResultMessage):
            session_id = msg.session_id
            # don't break — let iterator exhaust naturally to avoid SDK asyncio cancel-scope error

    for tc in tool_calls:
        tc[2] = "done"
    final_text = [{"role": "assistant", "content": text}] if text else []
    if not tool_calls and not text:
        final_text = [{"role": "assistant", "content": "*(no response)*"}]
    history = history + [tool_msg(t, c, s) for t, c, s in tool_calls] + final_text
    yield history, ""


def stop_handler():
    global _stop
    _stop = True


def new_chat():
    global session_id
    session_id = None
    return [], gr.update(samples=get_session_samples())


def on_session_select(evt: gr.SelectData):
    global session_id
    sessions = _read_sessions()
    idx = evt.index
    if idx >= len(sessions):
        return []
    sid = sessions[idx][1]
    session_id = sid
    return load_session_history(sid)


def refresh_sessions():
    return gr.update(samples=get_session_samples())


# ── UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="CC-UI") as demo:
    with gr.Sidebar(label="Sessions", width=240):
        gr.Markdown("**Recents**")
        session_list = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=get_session_samples(),
            label=None,
            headers=None,
            samples_per_page=50,
            type="index",
        )
        refresh_btn = gr.Button("↻ refresh", size="sm", variant="secondary")

    chatbot = gr.Chatbot(height=580, show_label=False, render_markdown=True, container=False, layout="panel", buttons=[], feedback_options=[])
    msg_box = gr.Textbox(
        placeholder="Message...", lines=1, show_label=False, submit_btn=True, container=False,
    )
    with gr.Row():
        mode_dd = gr.Dropdown(
            choices=PERMISSION_MODES, value="bypassPermissions",
            show_label=False, scale=0, min_width=170,
        )
        new_btn = gr.Button("＋ new", scale=0)
        stop_btn = gr.Button("⏹ stop", variant="stop", scale=0)

    msg_box.submit(send_message, inputs=[msg_box, mode_dd, chatbot], outputs=[chatbot, msg_box])
    stop_btn.click(stop_handler)
    new_btn.click(new_chat, outputs=[chatbot, session_list])
    session_list.select(on_session_select, outputs=[chatbot])
    refresh_btn.click(refresh_sessions, outputs=[session_list])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean(), footer_links=[])
