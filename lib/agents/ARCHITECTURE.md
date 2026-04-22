# The Agent Harness: A Deep Walkthrough

## The Problem This Solves

You're building a system where users can talk to AI agents — Claude Code, OpenCode, a plain chat LLM, maybe your own custom agent someday. Each of these agents works differently under the hood. Claude Code is a CLI that manages its own file editing and bash sessions. OpenCode is a different CLI with its own JSON event format. A plain ChatGPT-style agent just streams text back.

Without a unified layer, your backend has to know the intimate details of every agent. Your UI has to know them too. When you add a new agent, you're touching the backend task runner, the frontend event renderer, the database schema — everywhere. It's the kind of codebase where adding one thing breaks three others.

The agent harness exists to solve that. It gives every agent the same shape: same way to start a conversation, same stream of events coming back, same lifecycle. Your backend and UI only talk to the harness. The harness talks to the agents. Adding a new agent means writing one Python class. Nothing else changes.

This is the same pattern VS Code's Copilot uses internally. They call it `IAgent` / `AgentService`. We've translated the core ideas to Python, stripped out the VS Code-specific parts, and kept what matters.

---

## The Architecture, Bottom-Up

Let's build the mental model from the smallest pieces up. Each layer depends only on the one below it.

### Layer 1: Events

When an agent is working, things happen — it produces text, it calls a tool, it reports how many tokens it used, it hits an error. We need a common vocabulary for all of these.

An **event** is a small, immutable data object that describes one thing that happened. "Immutable" means once it's created, you can't change it. This matters because events get passed around, serialized, logged — you don't want something mutating them mid-flight.

```python
@dataclass(frozen=True)
class DeltaEvent:
    """A chunk of streaming text from the agent."""
    type: EventType = field(default=EventType.DELTA, init=False)
    content: str = ""
    message_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)
```

The `frozen=True` is what makes it immutable. The `type` field is set automatically — you can't accidentally create a `DeltaEvent` with `type=TOOL_START`.

Here are all the event types:

| Event | What it means |
|---|---|
| `DeltaEvent` | A piece of streaming text (the agent is "typing") |
| `MessageEvent` | A complete message (used less often than Delta) |
| `ToolStartEvent` | The agent started calling a tool (like "edit file" or "run bash") |
| `ToolCompleteEvent` | That tool call finished, here's the result |
| `ReasoningEvent` | The agent is thinking/planning (like Claude's extended thinking) |
| `UsageEvent` | Token counts and cost for this step |
| `ErrorEvent` | Something went wrong |
| `IdleEvent` | Turn is done, agent is ready for the next message |
| `SubagentStartEvent` | The agent spawned a sub-agent to handle part of the work |
| `SubagentEndEvent` | That sub-agent finished |

All events serialize cleanly to JSON via `event_to_dict()` and come back via `event_from_dict()`. This matters for the API layer — the backend streams these to the frontend as JSON.

The union type `AgentEvent` is just Python's way of saying "any one of these event types." It lets your code accept any event without listing all eleven types.

**What events can't do:** They don't carry binary data. If a tool produces a large file diff, the event carries a truncated string preview, not the full output. Events are status signals, not data transport.

### Layer 2: Sessions and Turns

A **session** is one conversation with one agent. Think of it like a chat thread. It has an ID, it knows which agent it belongs to, which model to use, and it keeps a history of what was said.

Inside a session, each user message and the agent's response to it is called a **turn**. You say something, the agent responds — that's one turn. The turn collects everything the agent did: the text it produced, the tools it called, how many tokens it used.

```
Session (one conversation)
  ├─ Turn 1: user says "fix the bug in auth.py"
  │    ├─ ResponsePart: reasoning ("let me look at the file...")
  │    ├─ ResponsePart: tool_call (view auth.py)
  │    ├─ ResponsePart: tool_call (edit auth.py)
  │    └─ ResponsePart: markdown ("Fixed the null check on line 42")
  ├─ Turn 2: user says "now add a test for it"
  │    └─ ...
  └─ state: IDLE (waiting for next message)
```

The session has a state machine:

```
CREATING → READY → ACTIVE → IDLE → ACTIVE → IDLE → ... → COMPLETED
                      ↓
                    ERROR
```

`ACTIVE` means the agent is processing a turn right now. `IDLE` means it finished and is waiting. This state machine prevents you from sending a second message while the first is still being processed — a real problem that causes race conditions if you don't guard against it.

The `Turn` object is mutable while it's active (text gets appended as it streams in), but once the turn completes, it's moved into the session's history list and effectively sealed.

**What sessions can't do:** They don't persist automatically. The session lives in memory. If the server restarts, in-progress sessions are lost. The backend separately saves task state to SQLite — the session is the in-flight representation, the database is the durable one.

### Layer 3: Tools

A **tool** is a capability the agent can invoke — read a file, run a shell command, search code, create a file. In the OpenAI API world, tools are defined by a JSON Schema that says "this tool is called `edit`, it takes `path`, `old_str`, and `new_str`."

The **ToolRegistry** is a global, thread-safe store of all available tools. You register a tool once, and any agent can use it.

The interesting part is `fn_to_tool()`. You write a normal Python function with type hints and a docstring:

```python
def edit(path: str, old_str: str, new_str: str) -> dict:
    """Replace one occurrence of old_str with new_str in the file.
    
    Args:
        path: Absolute path to the file to edit
        old_str: The exact string to find and replace
        new_str: The replacement string
    """
    # ... implementation ...
```

And `fn_to_tool()` turns it into a `ToolDefinition` with the correct JSON Schema by inspecting the function's signature and parsing the docstring. No manual schema writing. This is how the existing `ai.py` works, and it's genuinely good — we kept it.

Tools have **tags** (like `"file"`, `"code"`, `"bash"`) so agents can say "give me all tools tagged 'file'" instead of listing them by name.

**What the ToolRegistry can't do:** It doesn't handle approval flows. In VS Code Copilot, some tools require user confirmation before running (like deleting files). Our registry has a `requires_approval` field on the definition, but the approval UI isn't wired up yet. The agent or backend would need to check that flag and pause for confirmation.

### Layer 4: BaseAgent

This is the contract every agent must fulfill. It's an abstract class — you can't use it directly, you have to subclass it and fill in the blanks.

```python
class BaseAgent(ABC):
    @property
    def id(self) -> str: ...           # "claude-code", "chat", etc.
    @property
    def name(self) -> str: ...         # "Claude Code", "Chat", etc.
    
    async def create_session(self, model, cwd, config) -> AgentSession: ...
    async def send_message(self, session, message) -> AsyncIterator[AgentEvent]: ...
    async def stop(self, session) -> None: ...
```

The key design decision: `send_message()` is an **async generator**. That means it yields events one at a time as they happen, instead of collecting them all and returning at the end. This is what makes streaming work. The backend can forward each event to the frontend the instant it arrives.

There's a deliberate asymmetry here. `create_session()` returns a session object (a single thing). `send_message()` yields a stream of events (many things over time). This matches reality — creating a session is instant, but processing a message takes seconds to minutes.

Each agent also declares its capabilities:

```python
@property
def supports_tools(self) -> bool: return False

@property  
def supports_subagents(self) -> bool: return False
```

The UI uses these to decide what to show. A chat agent doesn't support tools, so there's no point rendering tool call UI for it.

**What BaseAgent can't do:** It doesn't enforce that your agent actually streams. You could write a `send_message()` that collects everything and yields it all at the end. It would work, but the user would see nothing until it's all done. The contract is a suggestion — Python can't enforce "yield frequently."

### Layer 5: AgentService

This is the orchestrator. It's the only thing the backend talks to.

```python
svc = AgentService()
svc.register(CodingAgent())    # register once at startup
svc.register(ChatAgent())

# For each user request:
session = await svc.create_session("claude-code", model="sonnet")
async for event in svc.send_message(session.id, "Fix the bug"):
    # event is a DeltaEvent, ToolStartEvent, etc.
    forward_to_frontend(event)
```

The service does three things your agents shouldn't have to worry about:

First, **routing**. It maps session IDs to agents. When a message comes in for session `abc123`, the service knows that session belongs to `claude-code` and routes to the right agent.

Second, **state management**. As events stream back from the agent, the service updates the session's Turn object automatically. A `DeltaEvent` appends text. A `ToolStartEvent` adds a tool call record. A `UsageEvent` increments token counters. The agent doesn't have to manage any of this — it yields events, the service handles bookkeeping.

Third, **lifecycle**. Creating sessions, stopping them mid-turn, cleaning them up when done. The service tracks all active sessions and can shut them all down on server shutdown.

**What AgentService can't do:** It's single-node only. All sessions live in the memory of one Python process. If you need to run agents across multiple servers, you'd need to add a session store (Redis, database) and a message broker. That's a different scale of problem.

---

## The Built-in Agents

### CodingAgent

This wraps the existing CLI providers — Claude Code and OpenCode. These CLIs are opinionated: they manage their own tools, their own file editing, their own bash sessions. The `CodingAgent` doesn't try to reinvent that. It just translates.

The translation is mechanical. The existing provider emits `ProviderEvent` objects with types like `EventType.TEXT`, `EventType.TOOL_START`. The `CodingAgent._map_event()` method converts each one to the corresponding `AgentEvent`:

```
ProviderEvent(TEXT, "hello")        → DeltaEvent(content="hello")
ProviderEvent(TOOL_START, {title})  → ToolStartEvent(tool_name=title)
ProviderEvent(COST, {usage})        → UsageEvent(input_tokens=..., ...)
```

This is a **bridge pattern** — the agent is just a translator between two interfaces. The actual intelligence is in the CLI provider.

One subtlety: the `CodingAgent` captures `session_id` from `COST` events and stores it in the session config. This is how session resumption works — when you send a follow-up message, the agent passes that session ID back to the CLI so it picks up where it left off.

### ChatAgent

The simplest agent. No tools, no sub-agents, no CLI wrapping. It opens a streaming connection to any OpenAI-compatible API and yields `DeltaEvent` for each text chunk.

This exists because not every task needs an agentic coding assistant. Sometimes you want to brainstorm, or ask a question, or review code in conversation. A plain chat agent is lighter, faster, and cheaper.

### InHouseAgent

This wraps your existing `AIAgent` from `lab/src/ai/ai.py`. That framework has its own event model (`Text`, `ToolCall`, `ToolResult`, `StepResult`) and its own tool-calling loop. The `InHouseAgent` runs it in a thread (since `ai.py` is synchronous) and translates its events to `AgentEvent`.

This is the agent you'd use for custom tool-using workflows where you want full control over which tools are available and how they behave.

---

## How the UI Changed

Before, the chat input had four things to configure:

1. **Model picker** — which AI provider (Claude, OpenCode, etc.)
2. **Mode selector** — Autonomous, Accept Edits, Plan, Interactive
3. **Advisor picker** — optional second AI to review the first one's work
4. **File attach** + Send

That's too many decisions for a user who wants to get something done. The mode and advisor were power-user features that most people never changed from the defaults.

Now it's two choices:

1. **Model picker** — which AI provider and model variant
2. **Agent picker** — which agent (Coding, Chat, etc.)

The agent encapsulates the mode. A `CodingAgent` always runs autonomously. A `ChatAgent` is always conversational. The advisor pattern, if you want it back someday, would become its own agent that wraps another agent — not a separate UI control.

---

## How to Add a New Agent

Write a class that extends `BaseAgent`. Here's the minimal version:

```python
from lib.agents.base import BaseAgent
from lib.agents.events import DeltaEvent, AgentEvent
from lib.agents.session import AgentSession, UserMessage

class MyAgent(BaseAgent):
    @property
    def id(self) -> str:
        return "my-agent"
    
    @property
    def name(self) -> str:
        return "My Custom Agent"
    
    async def create_session(self, model="", cwd="", config=None) -> AgentSession:
        session = AgentSession(model=model, cwd=cwd, config=config or {})
        session.ready()
        return session
    
    async def send_message(self, session, message):
        # Do your thing, yield events as you go
        yield DeltaEvent(content=f"You said: {message.text}")
    
    async def stop(self, session):
        pass  # handle cancellation if needed
```

Register it in `backend_v2.py`'s `_register_agents()`:

```python
from lib.agents_builtin.my_agent import MyAgent
agent_service.register(MyAgent())
```

It shows up in the agent picker automatically via `GET /agents`.

---

## What This Can't Do (Yet)

There are real limitations to be honest about.

**No streaming tool progress.** When a tool is running (say, a 30-second build), the user sees "tool started" and then nothing until "tool complete." VS Code Copilot streams tool progress events — we don't yet.

**No approval workflow.** Some tools (delete file, run destructive commands) should ask the user "are you sure?" before executing. The `ToolDefinition` has a `requires_approval` flag but nothing checks it. The coding agents bypass this because their underlying CLIs handle permissions internally.

**No session persistence.** If the server restarts, in-memory sessions are gone. The SQLite database stores completed task history, but there's no way to reconstruct an `AgentSession` from the database. Adding this would mean serializing session state and replaying it on load.

**No multi-agent coordination.** The harness supports sub-agent *events* (SubagentStartEvent / SubagentEndEvent) but doesn't have a built-in way for agents to spawn other agents through the service. That orchestration still lives inside individual agents or providers.

**No rate limiting or queueing.** If ten users send messages simultaneously, ten agents run simultaneously. For a single-user tool this is fine. For a shared service, you'd need a job queue.

---

## File Map

```
lib/agents/
  __init__.py     — Public API, all exports
  events.py       — 11 event types, serialize/deserialize
  session.py      — AgentSession, Turn, UserMessage, ResponsePart
  base.py         — BaseAgent abstract class
  tools.py        — ToolRegistry, ToolDefinition, fn_to_tool()
  service.py      — AgentService orchestrator

lib/agents_builtin/
  coding.py       — CodingAgent (Claude Code / OpenCode bridge)
  chat.py         — ChatAgent (plain OpenAI-compatible streaming)
  inhouse.py      — InHouseAgent (wraps ai.py framework)
```

The dependency graph flows strictly downward: `service.py` depends on `base.py`, which depends on `events.py` and `session.py`. `tools.py` depends on `events.py`. Nothing circular, nothing surprising.
