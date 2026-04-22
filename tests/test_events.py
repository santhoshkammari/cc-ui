"""
Test 1: Event System — The foundation of the agent harness.

Demonstrates:
- Creating all event types
- Serialization (event → dict → JSON)
- Deserialization (dict → event)
- Event immutability (frozen dataclasses)
"""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.agents.events import (
    DeltaEvent, MessageEvent, ToolStartEvent, ToolCompleteEvent,
    ReasoningEvent, UsageEvent, ErrorEvent, IdleEvent,
    SubagentStartEvent, SubagentEndEvent,
    event_to_dict, event_from_dict, EventType,
)


def test_all_event_types_exist():
    """Every event type can be instantiated with defaults."""
    events = [
        DeltaEvent(content="Hello world"),
        MessageEvent(role="assistant", content="Full message"),
        ToolStartEvent(tool_call_id="tc1", tool_name="edit", display_name="✏️ Edit File"),
        ToolCompleteEvent(tool_call_id="tc1", tool_name="edit", success=True, result="ok"),
        ReasoningEvent(content="Let me think about this..."),
        UsageEvent(input_tokens=100, output_tokens=50, total_cost_usd=0.001),
        ErrorEvent(error_type="api_error", message="Rate limited"),
        IdleEvent(),
        SubagentStartEvent(agent_name="researcher", agent_id="sub-1"),
        SubagentEndEvent(agent_id="sub-1", success=True),
    ]
    print(f"  ✅ Created {len(events)} event types")
    for e in events:
        assert hasattr(e, "type")
        assert hasattr(e, "timestamp")
        assert isinstance(e.type, EventType)
    print(f"  ✅ All events have type and timestamp fields")
    return events


def test_serialization_roundtrip():
    """Events survive JSON serialization and come back identical."""
    original = DeltaEvent(content="streaming chunk #42")
    d = event_to_dict(original)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = event_from_dict(d2)

    assert restored.content == original.content
    assert restored.message_id == original.message_id
    assert restored.timestamp == original.timestamp
    assert restored.type == EventType.DELTA
    print(f"  ✅ DeltaEvent roundtrip: '{original.content}' → JSON → '{restored.content}'")


def test_all_types_roundtrip():
    """Every event type survives serialization."""
    events = test_all_event_types_exist()
    for e in events:
        d = event_to_dict(e)
        restored = event_from_dict(d)
        assert restored.type == e.type
    print(f"  ✅ All {len(events)} event types survive JSON roundtrip")


def test_immutability():
    """Frozen dataclasses reject mutation."""
    e = DeltaEvent(content="original")
    try:
        e.content = "modified"
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass
    print(f"  ✅ Events are immutable (mutation raises error)")


def test_event_dict_structure():
    """Serialized events have the expected JSON shape for API transport."""
    e = ToolStartEvent(tool_call_id="tc-99", tool_name="bash", display_name="🔧 Run Command", input_args='{"command":"ls"}')
    d = event_to_dict(e)
    assert d["type"] == "tool_start"
    assert d["tool_name"] == "bash"
    assert d["input_args"] == '{"command":"ls"}'
    print(f"  ✅ Serialized structure: type={d['type']}, tool_name={d['tool_name']}")


if __name__ == "__main__":
    print("\n🧪 Test 1: Event System")
    print("-" * 50)
    test_all_event_types_exist()
    test_serialization_roundtrip()
    test_all_types_roundtrip()
    test_immutability()
    test_event_dict_structure()
    print("✅ All event tests passed\n")
