"""
Test 4: AgentService — Full orchestration with a real agent.

Demonstrates:
- Registering agents with the service
- Creating sessions
- Sending messages and consuming event streams
- Automatic Turn state updates from events
- Multi-turn conversations
- Session lifecycle (stop, dispose)
"""

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.agents.base import BaseAgent
from lib.agents.events import (
    AgentEvent, DeltaEvent, ToolStartEvent, ToolCompleteEvent,
    UsageEvent, IdleEvent, ErrorEvent,
)
from lib.agents.session import AgentSession, UserMessage, SessionState
from lib.agents.service import AgentService


# ── A mock agent for testing the service without real LLM calls ──

class EchoAgent(BaseAgent):
    """Agent that echoes back the user's message with a tool call."""

    @property
    def id(self) -> str:
        return "echo"

    @property
    def name(self) -> str:
        return "Echo Agent"

    @property
    def description(self) -> str:
        return "Echoes input with a simulated tool call"

    @property
    def icon(self) -> str:
        return "🔊"

    @property
    def supports_tools(self) -> bool:
        return True

    async def create_session(self, model="", cwd="", config=None) -> AgentSession:
        session = AgentSession(model=model or "echo-v1", cwd=cwd, config=config or {})
        session.ready()
        return session

    async def send_message(self, session, message):
        # Simulate: think → tool call → text response
        yield ToolStartEvent(
            tool_call_id="tc-echo-1",
            tool_name="echo_tool",
            display_name="🔊 Echo",
            input_args=f'{{"text": "{message.text}"}}',
        )
        await asyncio.sleep(0.01)  # Simulate work
        yield ToolCompleteEvent(
            tool_call_id="tc-echo-1",
            tool_name="echo_tool",
            success=True,
            result=f"Echoed: {message.text}",
        )
        yield DeltaEvent(content=f"You said: {message.text}")
        yield UsageEvent(input_tokens=10, output_tokens=5, total_cost_usd=0.0001)

    async def stop(self, session):
        pass


class FailAgent(BaseAgent):
    """Agent that always errors — for testing error handling."""

    @property
    def id(self) -> str:
        return "fail"

    @property
    def name(self) -> str:
        return "Fail Agent"

    async def create_session(self, model="", cwd="", config=None) -> AgentSession:
        session = AgentSession(model=model or "fail-v1", cwd=cwd, config=config or {})
        session.ready()
        return session

    async def send_message(self, session, message):
        raise RuntimeError("Intentional failure for testing")
        yield  # make it a generator

    async def stop(self, session):
        pass


async def test_register_and_list():
    """Register agents and list them."""
    svc = AgentService()
    svc.register(EchoAgent())
    svc.register(FailAgent())

    agents = svc.list_agents()
    assert len(agents) == 2
    assert agents[0]["id"] == "echo"
    assert agents[1]["id"] == "fail"
    print(f"  ✅ Registered {len(agents)} agents: {[a['id'] for a in agents]}")
    assert svc.default_agent_id == "echo"
    print(f"  ✅ Default agent: {svc.default_agent_id}")
    return svc


async def test_create_session():
    """Create a session through the service."""
    svc = await test_register_and_list()

    session = await svc.create_session("echo", model="echo-v1", cwd="/tmp")
    assert session.state == SessionState.READY
    assert session.agent_id == "echo"
    assert session.model == "echo-v1"
    print(f"  ✅ Session created: id={session.id}, agent={session.agent_id}, state={session.state.value}")
    return svc, session


async def test_send_message_and_consume_events():
    """Send a message, collect all events, verify Turn state was updated."""
    svc, session = await test_create_session()

    events = []
    async for event in svc.send_message(session.id, "Hello agent!"):
        events.append(event)

    event_types = [type(e).__name__ for e in events]
    print(f"  ✅ Received {len(events)} events: {event_types}")

    # Check events
    assert any(isinstance(e, ToolStartEvent) for e in events)
    assert any(isinstance(e, ToolCompleteEvent) for e in events)
    assert any(isinstance(e, DeltaEvent) for e in events)
    assert any(isinstance(e, UsageEvent) for e in events)
    assert any(isinstance(e, IdleEvent) for e in events)
    print(f"  ✅ All expected event types present")

    # Check session state was updated automatically
    assert session.state == SessionState.IDLE
    assert len(session.turns) == 1
    turn = session.turns[0]
    assert turn.answer == "You said: Hello agent!"
    assert turn.usage is not None
    assert turn.usage.input_tokens == 10
    print(f"  ✅ Turn auto-updated: answer='{turn.answer}', tokens={turn.usage.input_tokens}in/{turn.usage.output_tokens}out")

    # Check tool call was tracked
    tool_parts = [p for p in turn.response_parts if p.kind == "tool_call"]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_call.status == "complete"
    assert tool_parts[0].tool_call.success is True
    print(f"  ✅ Tool call tracked: {tool_parts[0].tool_call.tool_name} → {tool_parts[0].tool_call.status}")
    return svc, session


async def test_multi_turn():
    """Multiple turns in the same session build history."""
    svc, session = await test_send_message_and_consume_events()

    # Send second message
    events2 = []
    async for event in svc.send_message(session.id, "Second message"):
        events2.append(event)

    assert len(session.turns) == 2
    assert session.turns[1].answer == "You said: Second message"
    history = session.build_history()
    assert len(history) == 4  # 2 user + 2 assistant
    print(f"  ✅ Multi-turn: {len(session.turns)} turns, {len(history)} history messages")

    total = session.total_usage
    assert total.input_tokens == 20
    assert total.output_tokens == 10
    print(f"  ✅ Aggregated usage: {total.input_tokens}in / {total.output_tokens}out / ${total.total_cost_usd:.4f}")


async def test_error_handling():
    """Agent errors are caught and turned into ErrorEvent + IdleEvent."""
    svc = AgentService()
    svc.register(FailAgent())

    session = await svc.create_session("fail")
    events = []
    async for event in svc.send_message(session.id, "This will fail"):
        events.append(event)

    event_types = [type(e).__name__ for e in events]
    assert "ErrorEvent" in event_types
    assert "IdleEvent" in event_types
    print(f"  ✅ Error handled gracefully: {event_types}")
    assert session.state == SessionState.IDLE  # not stuck in ACTIVE
    print(f"  ✅ Session state recovered to: {session.state.value}")


async def test_dispose_session():
    """Sessions can be disposed (cleaned up)."""
    svc = AgentService()
    svc.register(EchoAgent())

    session = await svc.create_session("echo")
    sid = session.id
    assert svc.get_session(sid) is not None

    await svc.dispose_session(sid)
    assert svc.get_session(sid) is None
    assert session.state == SessionState.COMPLETED
    print(f"  ✅ Session disposed: state={session.state.value}, lookup returns None")


async def test_invalid_session():
    """Sending to a nonexistent session yields an error event."""
    svc = AgentService()
    svc.register(EchoAgent())

    events = []
    async for event in svc.send_message("nonexistent-id", "Hello"):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "not found" in events[0].message
    print(f"  ✅ Invalid session: {events[0].message}")


async def run_all():
    await test_register_and_list()
    await test_create_session()
    await test_send_message_and_consume_events()
    await test_multi_turn()
    await test_error_handling()
    await test_dispose_session()
    await test_invalid_session()


if __name__ == "__main__":
    print("\n🧪 Test 4: AgentService Orchestration")
    print("-" * 50)
    asyncio.run(run_all())
    print("✅ All service tests passed\n")
