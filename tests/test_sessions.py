"""
Test 2: Session & Turn Lifecycle — Conversation state management.

Demonstrates:
- Session state machine (creating → ready → active → idle → completed)
- Turn creation, text accumulation, tool call tracking
- Multi-turn conversations with history building
- Usage aggregation across turns
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.agents.session import (
    AgentSession, Turn, UserMessage, Attachment,
    ToolCallState, ResponsePart, UsageInfo,
    SessionState, TurnState,
)


def test_session_lifecycle():
    """Walk through the full session state machine."""
    session = AgentSession(agent_id="test", model="qwen3.5-9b")

    assert session.state == SessionState.CREATING
    print(f"  ✅ Initial state: {session.state.value}")

    session.ready()
    assert session.state == SessionState.READY
    print(f"  ✅ After ready(): {session.state.value}")

    turn = session.start_turn(UserMessage(text="Hello"))
    assert session.state == SessionState.ACTIVE
    assert session.active_turn is turn
    print(f"  ✅ After start_turn(): {session.state.value}, active_turn exists")

    session.complete_turn()
    assert session.state == SessionState.IDLE
    assert session.active_turn is None
    assert len(session.turns) == 1
    print(f"  ✅ After complete_turn(): {session.state.value}, {len(session.turns)} turn(s) in history")

    session.finish()
    assert session.state == SessionState.COMPLETED
    print(f"  ✅ After finish(): {session.state.value}")


def test_turn_text_accumulation():
    """Text chunks accumulate into a single response, like streaming."""
    turn = Turn(user_message=UserMessage(text="Tell me about Python"))

    turn.add_text("Python is ")
    turn.add_text("a programming language ")
    turn.add_text("created by Guido van Rossum.")

    assert turn.answer == "Python is a programming language created by Guido van Rossum."
    assert len(turn.response_parts) == 1  # All text merged into one part
    print(f"  ✅ 3 text chunks → 1 response part: '{turn.answer[:50]}...'")


def test_turn_mixed_content():
    """Turns can have text, reasoning, and tool calls interleaved."""
    turn = Turn(user_message=UserMessage(text="Fix the bug"))

    turn.add_reasoning("I need to look at the error first")
    turn.add_tool_call(ToolCallState(
        tool_call_id="tc1", tool_name="view",
        display_name="📄 View File", status="running",
        input_args='{"path": "/app/main.py"}',
    ))
    turn.add_text("I found the issue — ")
    turn.add_tool_call(ToolCallState(
        tool_call_id="tc2", tool_name="edit",
        display_name="✏️ Edit File", status="running",
    ))
    turn.add_text("Fixed the null pointer on line 42.")

    parts = turn.response_parts
    assert parts[0].kind == "reasoning"
    assert parts[1].kind == "tool_call"
    assert parts[2].kind == "markdown"
    assert parts[3].kind == "tool_call"
    assert parts[4].kind == "markdown"
    print(f"  ✅ Mixed content: {[p.kind for p in parts]}")
    print(f"  ✅ Final answer: '{turn.answer}'")


def test_multi_turn_history():
    """Session builds OpenAI-style message history from completed turns."""
    session = AgentSession(agent_id="test", model="test-model")
    session.ready()

    # Turn 1
    t1 = session.start_turn(UserMessage(text="What is 2+2?"))
    t1.add_text("4")
    session.complete_turn()

    # Turn 2
    t2 = session.start_turn(UserMessage(text="And 3+3?"))
    t2.add_text("6")
    session.complete_turn()

    history = session.build_history()
    assert len(history) == 4  # user, assistant, user, assistant
    assert history[0] == {"role": "user", "content": "What is 2+2?"}
    assert history[1] == {"role": "assistant", "content": "4"}
    assert history[2] == {"role": "user", "content": "And 3+3?"}
    assert history[3] == {"role": "assistant", "content": "6"}
    print(f"  ✅ 2 turns → {len(history)} history messages")
    for m in history:
        print(f"      {m['role']:>10}: {m['content']}")


def test_usage_aggregation():
    """Usage stats aggregate across turns."""
    session = AgentSession(agent_id="test", model="test-model")
    session.ready()

    t1 = session.start_turn(UserMessage(text="Turn 1"))
    t1.usage = UsageInfo(input_tokens=100, output_tokens=50, total_cost_usd=0.001)
    t1.add_text("Response 1")
    session.complete_turn()

    t2 = session.start_turn(UserMessage(text="Turn 2"))
    t2.usage = UsageInfo(input_tokens=200, output_tokens=80, total_cost_usd=0.002)
    t2.add_text("Response 2")
    session.complete_turn()

    total = session.total_usage
    assert total.input_tokens == 300
    assert total.output_tokens == 130
    assert abs(total.total_cost_usd - 0.003) < 1e-9
    print(f"  ✅ Aggregated: {total.input_tokens} in + {total.output_tokens} out = ${total.total_cost_usd:.4f}")


def test_error_and_cancel():
    """Turns can fail or be cancelled."""
    session = AgentSession(agent_id="test", model="test")
    session.ready()

    # Error turn
    session.start_turn(UserMessage(text="Bad request"))
    session.fail_turn("API returned 500")
    assert session.turns[0].state == TurnState.ERROR
    assert session.turns[0].error == "API returned 500"
    print(f"  ✅ Failed turn: state={session.turns[0].state.value}, error='{session.turns[0].error}'")

    # Cancelled turn
    session.start_turn(UserMessage(text="Stop this"))
    session.cancel_turn()
    assert session.turns[1].state == TurnState.CANCELLED
    print(f"  ✅ Cancelled turn: state={session.turns[1].state.value}")


def test_attachments():
    """User messages can carry file attachments."""
    msg = UserMessage(
        text="Review this file",
        attachments=[
            Attachment(type="file", path="/app/main.py", display_name="main.py"),
            Attachment(type="image", path="/tmp/screenshot.png", mime_type="image/png"),
        ],
    )
    d = msg.to_dict()
    assert len(d["attachments"]) == 2
    assert d["attachments"][0]["path"] == "/app/main.py"
    print(f"  ✅ UserMessage with {len(msg.attachments)} attachments serialized")


def test_session_serialization():
    """Full session serializes to dict for API responses."""
    session = AgentSession(agent_id="test", model="qwen3.5-9b", cwd="/app")
    session.ready()
    t = session.start_turn(UserMessage(text="Hi"))
    t.add_text("Hello!")
    session.complete_turn()

    d = session.to_dict()
    assert d["agent_id"] == "test"
    assert d["model"] == "qwen3.5-9b"
    assert len(d["turns"]) == 1
    assert d["state"] == "idle"
    print(f"  ✅ Session serialized: agent={d['agent_id']}, model={d['model']}, state={d['state']}")


if __name__ == "__main__":
    print("\n🧪 Test 2: Session & Turn Lifecycle")
    print("-" * 50)
    test_session_lifecycle()
    test_turn_text_accumulation()
    test_turn_mixed_content()
    test_multi_turn_history()
    test_usage_aggregation()
    test_error_and_cancel()
    test_attachments()
    test_session_serialization()
    print("✅ All session tests passed\n")
