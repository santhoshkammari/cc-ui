"""
Agent orchestration: launch_task, read_agent, list_agents, execute_skill.

Implements a hierarchical manager-worker pattern where a main agent delegates
tasks to isolated sub-agents.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    EXPLORE = "explore"
    TASK = "task"
    GENERAL_PURPOSE = "general-purpose"
    CODE_REVIEW = "code-review"


class AgentStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


DEFAULT_MODELS: dict[AgentType, str] = {
    AgentType.EXPLORE: "claude-haiku-4.5",
    AgentType.TASK: "claude-haiku-4.5",
    AgentType.GENERAL_PURPOSE: "claude-sonnet-4.6",
    AgentType.CODE_REVIEW: "claude-sonnet-4.6",
}


# ---------------------------------------------------------------------------
# Agent data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_index: int
    role: str
    content: str


@dataclass
class Agent:
    agent_id: str
    name: str
    agent_type: AgentType
    description: str
    model: str
    prompt: str
    status: AgentStatus = AgentStatus.RUNNING
    turns: list[Turn] = field(default_factory=list)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _handler: Optional[Callable[..., str]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# AgentManager — singleton registry
# ---------------------------------------------------------------------------

class AgentManager:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = threading.Lock()

    def register(self, agent: Agent) -> None:
        with self._lock:
            self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def all_agents(self, *, include_completed: bool = True) -> list[Agent]:
        agents = list(self._agents.values())
        if not include_completed:
            agents = [
                a for a in agents
                if a.status in (AgentStatus.RUNNING, AgentStatus.IDLE)
            ]
        return agents


_manager = AgentManager()


# ---------------------------------------------------------------------------
# Default handler (stub — replace with real LLM call in production)
# ---------------------------------------------------------------------------

def _default_handler(agent: Agent) -> None:
    """Placeholder that simulates agent work.  Override in production."""
    agent.turns.append(
        Turn(
            turn_index=0,
            role="assistant",
            content=(
                f"[stub] Agent '{agent.name}' ({agent.agent_type.value}) "
                f"received prompt: {agent.prompt[:200]}..."
            ),
        )
    )
    agent.status = AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch_task(
    name: str,
    prompt: str,
    agent_type: str,
    description: str,
    *,
    model: Optional[str] = None,
    mode: str = "background",
    handler: Optional[Callable[..., None]] = None,
) -> dict:
    """Launch a specialised sub-agent.

    Parameters
    ----------
    name:
        Short human-readable name (e.g. ``"auth-flow"``).
    prompt:
        Complete task description with all context.
    agent_type:
        ``"explore"`` | ``"task"`` | ``"general-purpose"`` | ``"code-review"``.
    description:
        3-5 word UI summary.
    model:
        Override the default model for the agent type.
    mode:
        ``"background"`` (default) or ``"sync"``.
    handler:
        Custom callable ``(Agent) -> None``.  Defaults to a stub.

    Returns
    -------
    dict with ``agent_id`` and ``status``.
    """
    atype = AgentType(agent_type)
    agent_id = f"{name}-{uuid.uuid4().hex[:4]}"

    agent = Agent(
        agent_id=agent_id,
        name=name,
        agent_type=atype,
        description=description,
        model=model or DEFAULT_MODELS[atype],
        prompt=prompt,
    )
    _manager.register(agent)

    work_fn = handler or _default_handler

    if mode == "sync":
        try:
            work_fn(agent)
        except Exception as exc:
            agent.status = AgentStatus.FAILED
            agent.turns.append(
                Turn(turn_index=len(agent.turns), role="error", content=str(exc))
            )
        return {"agent_id": agent_id, "status": agent.status.value}

    # background
    def _run() -> None:
        try:
            work_fn(agent)
        except Exception as exc:
            agent.status = AgentStatus.FAILED
            agent.turns.append(
                Turn(turn_index=len(agent.turns), role="error", content=str(exc))
            )

    t = threading.Thread(target=_run, daemon=True)
    agent._thread = t
    t.start()

    return {"agent_id": agent_id, "status": "running"}


def read_agent(
    agent_id: str,
    *,
    wait: bool = False,
    timeout: float = 30,
    since_turn: Optional[int] = None,
) -> dict:
    """Read status and results from a background agent.

    Parameters
    ----------
    agent_id:
        ID returned by :func:`launch_task`.
    wait:
        Block until the agent completes.
    timeout:
        Max seconds to wait (only with ``wait=True``).  Max 180.
    since_turn:
        Only return turns after this index (exclusive).
    """
    agent = _manager.get(agent_id)
    if agent is None:
        raise KeyError(f"No agent with id={agent_id!r}")

    timeout = min(timeout, 180)

    if wait and agent._thread is not None:
        agent._thread.join(timeout=timeout)

    turns = agent.turns
    if since_turn is not None:
        turns = [t for t in turns if t.turn_index > since_turn]

    return {
        "agent_id": agent.agent_id,
        "status": agent.status.value,
        "turns": [
            {"turn_index": t.turn_index, "role": t.role, "content": t.content}
            for t in turns
        ],
    }


def list_agents(*, include_completed: bool = True) -> list[dict]:
    """List all agents.

    Parameters
    ----------
    include_completed:
        When ``False``, only return ``running`` and ``idle`` agents.
    """
    return [
        {
            "agent_id": a.agent_id,
            "name": a.name,
            "agent_type": a.agent_type.value,
            "status": a.status.value,
            "description": a.description,
        }
        for a in _manager.all_agents(include_completed=include_completed)
    ]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

# Skill registry — register custom skills with ``register_skill(name, fn)``
_skill_registry: dict[str, Callable[..., Any]] = {}


def register_skill(name: str, fn: Callable[..., Any]) -> None:
    """Register a skill handler for use with :func:`execute_skill`."""
    _skill_registry[name] = fn


def execute_skill(skill: str) -> Any:
    """Execute a named skill in the main conversation context.

    Parameters
    ----------
    skill:
        Skill name (e.g. ``"pdf"``, ``"xlsx"``, ``"customize-cloud-agent"``).

    Returns
    -------
    Whatever the skill handler returns.

    Raises
    ------
    KeyError
        If the skill is not registered.
    """
    fn = _skill_registry.get(skill)
    if fn is None:
        raise KeyError(
            f"Skill {skill!r} is not registered. "
            f"Available: {list(_skill_registry.keys())}"
        )
    return fn()
