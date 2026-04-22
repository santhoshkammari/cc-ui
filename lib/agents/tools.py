"""
Unified AI Agent Harness — Tool Registry

Singleton registry for tools. Agents opt-in to which tools they support.
Inspired by VS Code Copilot ToolRegistry + existing fn_to_tool() from ai.py.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .events import ToolCompleteEvent, ToolStartEvent


# ─── Tool Definition ──────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Schema + handler for a single tool."""
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema object
    handler: Callable | None = None
    requires_approval: bool = False
    tags: list[str] = field(default_factory=list)  # e.g. ["file", "code", "web"]

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "tags": self.tags,
        }


# ─── Tool Result ──────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of invoking a tool."""
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    error: str | None = None


# ─── fn_to_tool: auto-generate schema from function ──────────────────

_PY_TO_JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """Parse 'Args:' section from Google-style docstrings."""
    descriptions: dict[str, str] = {}
    if not docstring:
        return descriptions
    in_args = False
    current_param = None
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                break  # Left the Args section
            # New param: "  name (type): description" or "  name: description"
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if m:
                current_param = m.group(1)
                descriptions[current_param] = m.group(2).strip()
            elif current_param and stripped:
                descriptions[current_param] += " " + stripped
    return descriptions


def fn_to_tool(fn: Callable, *, name: str | None = None, tags: list[str] | None = None) -> ToolDefinition:
    """
    Introspect a Python function's signature + docstring to produce a ToolDefinition.
    Mirrors ai.py's fn_to_tool but returns a proper ToolDefinition.
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    # First line of docstring is the description
    desc_lines = doc.split("\n")
    description = desc_lines[0].strip() if desc_lines else fn.__name__

    param_docs = _parse_docstring_args(doc)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        # Determine JSON type from annotation
        annotation = param.annotation
        json_type = "string"
        if annotation != inspect.Parameter.empty:
            type_name = getattr(annotation, "__name__", str(annotation))
            # Handle Optional[X]
            origin = getattr(annotation, "__origin__", None)
            if origin is not None:
                args = getattr(annotation, "__args__", ())
                if args:
                    type_name = getattr(args[0], "__name__", str(args[0]))
            json_type = _PY_TO_JSON.get(type_name, "string")

        prop: dict[str, Any] = {"type": json_type}
        if pname in param_docs:
            prop["description"] = param_docs[pname]

        # Default value
        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop["default"] = param.default
        else:
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                required.append(pname)

        properties[pname] = prop

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return ToolDefinition(
        name=name or fn.__name__,
        description=description,
        parameters=parameters,
        handler=fn,
        tags=tags or [],
    )


# ─── Tool Registry (Singleton) ───────────────────────────────────────

class ToolRegistry:
    """
    Global, thread-safe tool registry.
    Tools are registered once and agents opt-in by name or tag.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._lock = threading.Lock()

    def register(self, tool: ToolDefinition) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def register_function(
        self,
        fn: Callable,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> ToolDefinition:
        """Convenience: auto-generate ToolDefinition from a function and register it."""
        td = fn_to_tool(fn, name=name, tags=tags)
        self.register(td)
        return td

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_tools(
        self,
        names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """
        Get tools filtered by names and/or tags.
        If both None, returns all tools.
        """
        with self._lock:
            tools = list(self._tools.values())

        if names is not None:
            name_set = set(names)
            tools = [t for t in tools if t.name in name_set]

        if tags is not None:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set & set(t.tags)]

        return tools

    def get_openai_schemas(
        self,
        names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Get OpenAI function-calling schemas for filtered tools."""
        return [t.to_openai_schema() for t in self.get_tools(names=names, tags=tags)]

    async def invoke(
        self,
        name: str,
        arguments: str | dict,
        tool_call_id: str = "",
    ) -> ToolResult:
        """
        Invoke a tool by name. Handles both sync and async handlers.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                success=False,
                content="",
                error=f"Tool '{name}' not found",
            )
        if tool.handler is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                success=False,
                content="",
                error=f"Tool '{name}' has no handler",
            )

        # Parse arguments
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
        else:
            args = arguments

        try:
            result = tool.handler(**args)
            # Support async handlers
            if inspect.isawaitable(result):
                result = await result
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                success=True,
                content=content,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                success=False,
                content="",
                error=f"{type(e).__name__}: {e}",
            )

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ─── Global instance ─────────────────────────────────────────────────

tool_registry = ToolRegistry()
