# CC-UI Agent Harness — Test Results

## Environment

| Component | Details |
|-----------|---------|
| **vLLM Server** | `192.168.170.76:8000` |
| **Model** | `qwen3.5-9b` (path: `/home/ng6309/datascience/santhosh/models/qwen3.5-9b`) |
| **Python** | 3.x with `asyncio`, `openai` SDK |
| **Protocol** | OpenAI-compatible Chat Completions API |

## Summary

```
  ✅ Event System                             PASS  (0.07s)
  ✅ Session & Turn Lifecycle                 PASS  (0.08s)
  ✅ Tool Registry                            PASS  (0.08s)
  ✅ AgentService Orchestration               PASS  (0.12s)
  ✅ Live Integration (vLLM)                  PASS  (171.45s)

  Total: 5/5 passed in 171.80s
```

## Test Breakdown

### 1. Event System (`test_events.py`) — 0.07s

Tests the foundation: 10 frozen event dataclasses with serialization.

- Created all 10 event types (Delta, Message, ToolStart, ToolComplete, Reasoning, Usage, Error, Idle, Progress, Cancel)
- Every event auto-populates `type` and `timestamp` fields
- Full JSON roundtrip: serialize → deserialize → compare — all 10 types survive
- Immutability enforced: mutating a frozen event raises `FrozenInstanceError`
- Serialized structure verified (field names, nesting)

### 2. Session & Turn Lifecycle (`test_sessions.py`) — 0.08s

Tests the session state machine and turn accumulation.

- State transitions: `creating → ready → active → idle → completed`
- Text streaming: 3 delta chunks merge into one coherent response part
- Mixed content: reasoning + tool calls + markdown all coexist in a single turn
- Multi-turn history: 2 turns produce 4 OpenAI-style messages (user/assistant pairs)
- Usage aggregation: token counts sum correctly across turns
- Error recovery: failed turns set `state=error` with error message preserved
- Cancellation: cancelled turns set `state=cancelled`
- Attachments: `UserMessage` with file attachments serializes correctly

### 3. Tool Registry (`test_tools.py`) — 0.08s

Tests `fn_to_tool()` auto-schema generation and the registry singleton.

- `fn_to_tool(add_numbers)` → extracts name, description, parameters, types, required fields from signature + docstring
- Complex function with defaults and Optional types handled correctly
- OpenAI function-calling schema generated in correct format
- Registry: register, filter by tag, filter by name, get OpenAI schemas
- Sync invocation: `add_numbers(5, 3) = 8`
- Async invocation: `async_fetch(url)` works via `asyncio`
- Error handling: missing tool → graceful error, bad args → graceful error

### 4. AgentService Orchestration (`test_service.py`) — 0.12s

Tests the full orchestration loop with mock `EchoAgent` and `FailAgent`.

- Agent registration and default agent selection
- Session creation with correct initial state
- Full event stream: `ToolStart → ToolComplete → Delta → Usage → Idle`
- Auto-state updates: `_apply_event()` builds turn content from events without agent involvement
- Tool call tracking: tool name, arguments, and result stored on turn
- Multi-turn: 2 turns accumulate 4 history messages, usage aggregates
- Error handling: `FailAgent` produces `ErrorEvent`, session recovers to `idle`
- Session disposal: state moves to `completed`, lookup returns `None`
- Invalid session ID: raises clean error message

### 5. Live vLLM Integration (`test_live_vllm.py`) — 171.45s

Tests against a real vLLM server running `qwen3.5-9b`.

| Test | Events | Time | Tokens |
|------|--------|------|--------|
| Single turn ("What is 2+2?") | 230 | 8.96s | 22 in / 229 out |
| Multi-turn (2 turns) | 2673 | 51.79s | 63 in / 4389 out |
| Code generation (prime checker) | 3482 | 67.87s | 11063 chars |
| Concurrent sessions (2 parallel) | — | 4.40s | — |
| Session cleanup | — | instant | — |

Key observations:
- **Streaming works end-to-end**: ChatAgent → OpenAI SDK → vLLM → async event stream
- **Multi-turn context preserved**: second turn references first turn's answer
- **Code generation**: model produced a complete, documented `is_prime()` function (351 lines with thinking)
- **Concurrency**: two independent sessions stream simultaneously without interference
- **Cleanup**: session disposal and service shutdown are clean, no resource leaks

## Architecture Validated

This test suite proves the full stack works:

```
User prompt
  → AgentService.send_message()
    → BaseAgent.send_message()  (ChatAgent for live tests)
      → OpenAI AsyncClient streaming
        → vLLM server (qwen3.5-9b)
      ← SSE chunks
    ← AgentEvent stream (Delta, Usage, Idle)
  ← _apply_event() auto-updates Turn state
← Session has complete history, usage, state
```

Every layer — events, sessions, tools, agents, service, live LLM — is tested and working.
