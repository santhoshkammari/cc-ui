"""
Test 5: Live Integration — ChatAgent against a real vLLM server.

This test hits a live OpenAI-compatible vLLM endpoint to verify
the full stack works end-to-end: Agent → Session → Event Stream → Real LLM.

Server: 192.168.170.76:8000
Model: qwen3.5-9b (Qwen 3.5 9B)
"""

import asyncio
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.agents.events import DeltaEvent, UsageEvent, ErrorEvent, IdleEvent
from lib.agents.session import UserMessage, SessionState
from lib.agents.service import AgentService
from lib.agents_builtin.chat import ChatAgent

VLLM_BASE_URL = "http://192.168.170.76:8000/v1"
VLLM_MODEL = "/home/ng6309/datascience/santhosh/models/qwen3.5-9b"


async def test_single_turn():
    """Single message → streaming response from live vLLM."""
    svc = AgentService()
    svc.register(ChatAgent(
        agent_id="vllm-chat",
        agent_name="vLLM Chat",
        default_model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
    ))

    session = await svc.create_session("vllm-chat")
    assert session.state == SessionState.READY
    print(f"  ✅ Session created: model={session.model[:40]}...")

    events = []
    text_chunks = 0
    t0 = time.time()

    async for event in svc.send_message(session.id, "What is 2+2? Answer in one word."):
        events.append(event)
        if isinstance(event, DeltaEvent):
            text_chunks += 1

    elapsed = time.time() - t0
    event_types = [type(e).__name__ for e in events]

    print(f"  ✅ Received {len(events)} events in {elapsed:.2f}s: {set(event_types)}")
    print(f"  ✅ Streamed {text_chunks} text chunks")

    assert text_chunks > 0, "Expected at least one DeltaEvent"
    assert any(isinstance(e, IdleEvent) for e in events), "Expected IdleEvent"

    # Check session state
    assert session.state == SessionState.IDLE
    assert len(session.turns) == 1
    answer = session.turns[0].answer
    print(f"  ✅ Answer: '{answer.strip()[:100]}'")
    print(f"  ✅ Session: state={session.state.value}, turns={len(session.turns)}")

    # Usage
    if session.turns[0].usage:
        u = session.turns[0].usage
        print(f"  ✅ Usage: {u.input_tokens}in / {u.output_tokens}out")

    return svc, session


async def test_multi_turn_conversation():
    """Multi-turn conversation maintaining context."""
    svc, session = await test_single_turn()

    # Second turn — should remember context
    events2 = []
    async for event in svc.send_message(session.id, "Multiply that result by 10."):
        events2.append(event)

    assert session.state == SessionState.IDLE
    assert len(session.turns) == 2
    answer2 = session.turns[1].answer
    print(f"  ✅ Turn 2 answer: '{answer2.strip()[:100]}'")

    history = session.build_history()
    print(f"  ✅ Full history: {len(history)} messages")
    for m in history:
        preview = m["content"][:60].replace("\n", " ")
        print(f"      {m['role']:>10}: {preview}...")

    total = session.total_usage
    print(f"  ✅ Total usage: {total.input_tokens}in / {total.output_tokens}out")

    return svc, session


async def test_longer_generation():
    """Test with a prompt that requires more substantial output."""
    svc = AgentService()
    svc.register(ChatAgent(
        agent_id="vllm-chat",
        agent_name="vLLM Chat",
        default_model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
    ))

    session = await svc.create_session("vllm-chat")

    events = []
    t0 = time.time()
    async for event in svc.send_message(
        session.id,
        "Write a Python function that checks if a number is prime. Include a docstring. Keep it under 15 lines."
    ):
        events.append(event)

    elapsed = time.time() - t0
    answer = session.turns[0].answer
    delta_count = sum(1 for e in events if isinstance(e, DeltaEvent))

    print(f"  ✅ Generated {len(answer)} chars in {elapsed:.2f}s ({delta_count} chunks)")
    print(f"  ✅ Response preview:")
    for line in answer.strip().split("\n")[:12]:
        print(f"      {line}")
    if len(answer.strip().split("\n")) > 12:
        print(f"      ... ({len(answer.strip().split(chr(10)))} lines total)")

    assert len(answer) > 50, "Expected substantial response"
    assert delta_count > 5, "Expected streaming (multiple chunks)"


async def test_concurrent_sessions():
    """Two sessions running simultaneously against the same server."""
    svc = AgentService()
    svc.register(ChatAgent(
        agent_id="vllm-chat",
        agent_name="vLLM Chat",
        default_model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
    ))

    s1 = await svc.create_session("vllm-chat")
    s2 = await svc.create_session("vllm-chat")

    async def collect(sid, prompt):
        events = []
        async for e in svc.send_message(sid, prompt):
            events.append(e)
        return events

    t0 = time.time()
    r1, r2 = await asyncio.gather(
        collect(s1.id, "What is the capital of France? One word."),
        collect(s2.id, "What is the capital of Japan? One word."),
    )
    elapsed = time.time() - t0

    a1 = s1.turns[0].answer.strip()
    a2 = s2.turns[0].answer.strip()

    print(f"  ✅ Concurrent sessions completed in {elapsed:.2f}s")
    print(f"      Session 1: '{a1[:60]}'")
    print(f"      Session 2: '{a2[:60]}'")
    assert len(a1) > 0 and len(a2) > 0


async def test_session_cleanup():
    """Sessions dispose cleanly."""
    svc = AgentService()
    svc.register(ChatAgent(
        agent_id="vllm-chat",
        agent_name="vLLM Chat",
        default_model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
    ))

    session = await svc.create_session("vllm-chat")
    async for _ in svc.send_message(session.id, "Hi"):
        pass

    await svc.dispose_session(session.id)
    assert svc.get_session(session.id) is None
    assert session.state == SessionState.COMPLETED
    print(f"  ✅ Session disposed: state={session.state.value}")

    await svc.shutdown()
    print(f"  ✅ AgentService shutdown complete")


async def run_all():
    await test_single_turn()
    print()
    await test_multi_turn_conversation()
    print()
    await test_longer_generation()
    print()
    await test_concurrent_sessions()
    print()
    await test_session_cleanup()


if __name__ == "__main__":
    print("\n🧪 Test 5: Live Integration (vLLM @ 192.168.170.76)")
    print("-" * 50)
    asyncio.run(run_all())
    print("\n✅ All live integration tests passed\n")
