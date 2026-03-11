import os
import json
import gradio as gr
from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    AssistantMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,
)

os.environ.pop("CLAUDECODE", None)

PERMISSION_MODES = ["bypassPermissions", "acceptEdits", "dontAsk", "default", "plan"]
SKIP_SUBTYPES = {"init", "system"}  # noisy system messages to ignore

session_id: str | None = None
_stop = False




def make_options(mode: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(permission_mode=mode, resume=session_id)


def fmt_tool_use(block: ToolUseBlock) -> str:
    args = json.dumps(block.input, indent=2) if block.input else ""
    return f"**▶ {block.name}**\n```json\n{args}\n```"


def fmt_tool_result(block: ToolResultBlock) -> str:
    content = block.content or ""
    if isinstance(content, list):
        content = "\n".join(c.get("text", str(c)) for c in content)
    status = " ❌" if block.is_error else ""
    preview = str(content)[:400] + ("…" if len(str(content)) > 400 else "")
    return f"**◀ result{status}**\n```\n{preview}\n```"


async def send_message(prompt: str, mode: str, history: list):
    global session_id, _stop
    if not prompt.strip():
        yield history, ""
        return

    _stop = False
    history = history + [{"role": "user", "content": prompt}]
    yield history, ""

    text = ""
    extras = []

    async for msg in query(prompt=prompt, options=make_options(mode)):
        if _stop:
            extras.append("⏹ *stopped*")
            break

        if isinstance(msg, AssistantMessage):
            if msg.error:
                history = history + [{"role": "assistant", "content": f"❌ {msg.error}"}]
                yield history, ""
                return
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text += block.text
                elif isinstance(block, ThinkingBlock):
                    extras.append(f"💭 *thinking…*\n> {block.thinking[:200]}")
                elif isinstance(block, ToolUseBlock):
                    extras.append(fmt_tool_use(block))
                elif isinstance(block, ToolResultBlock):
                    extras.append(fmt_tool_result(block))
            combined = "\n\n---\n\n".join(extras + ([text] if text else []))
            yield history + [{"role": "assistant", "content": combined}], ""

        elif isinstance(msg, SystemMessage):
            if msg.subtype not in SKIP_SUBTYPES:
                desc = getattr(msg, "description", None) or msg.subtype
                extras.append(f"⚙ *{desc}*")
                combined = "\n\n---\n\n".join(extras + ([text] if text else []))
                yield history + [{"role": "assistant", "content": combined}], ""

        elif isinstance(msg, ResultMessage):
            session_id = msg.session_id
            break

    parts = extras + ([text] if text else [])
    final = "\n\n---\n\n".join(parts) if parts else "*(no response)*"
    history = history + [{"role": "assistant", "content": final}]
    yield history, ""


def stop_handler():
    global _stop
    _stop = True


def new_session():
    global session_id
    session_id = None
    return []


with gr.Blocks(title="CC-UI") as demo:
    chatbot = gr.Chatbot(height=600, show_label=False, render_markdown=True)
    msg_box = gr.Textbox(
        placeholder="Message...",
        lines=1, show_label=False, submit_btn=True,
    )
    with gr.Row():
        mode_dd = gr.Dropdown(
            choices=PERMISSION_MODES, value="bypassPermissions",
            show_label=False, scale=0, min_width=180,
        )
        stop_btn = gr.Button("⏹ stop", variant="stop", scale=0)
        new_btn = gr.Button("+ new", scale=0)

    msg_box.submit(send_message, inputs=[msg_box, mode_dd, chatbot], outputs=[chatbot, msg_box])
    stop_btn.click(stop_handler)
    new_btn.click(new_session, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean())
