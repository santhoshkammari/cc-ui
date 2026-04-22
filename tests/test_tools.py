"""
Test 3: Tool Registry — Auto-schema generation and invocation.

Demonstrates:
- fn_to_tool(): Python function → OpenAI tool schema (automatic)
- ToolRegistry: register, lookup, filter by tags
- Tool invocation (sync and async handlers)
- Schema correctness for the OpenAI function-calling API
"""

import asyncio
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.agents.tools import (
    ToolRegistry, ToolDefinition, ToolResult, fn_to_tool, tool_registry,
)


# ── Sample tools to register ──

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b


def search_files(pattern: str, path: str = ".", case_sensitive: bool = True) -> list:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern to match (e.g. '*.py')
        path: Directory to search in
        case_sensitive: Whether the search is case-sensitive
    """
    return [f"{path}/{pattern.replace('*', 'example')}"]


async def async_fetch(url: str) -> str:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch
    """
    return f"<html>Content from {url}</html>"


def test_fn_to_tool_basic():
    """Auto-generate tool schema from a simple function."""
    td = fn_to_tool(add_numbers)

    assert td.name == "add_numbers"
    assert td.description == "Add two numbers together."
    assert "a" in td.parameters["properties"]
    assert "b" in td.parameters["properties"]
    assert td.parameters["properties"]["a"]["type"] == "integer"
    assert td.parameters["required"] == ["a", "b"]
    print(f"  ✅ fn_to_tool(add_numbers): name={td.name}, params={list(td.parameters['properties'].keys())}")
    print(f"      required={td.parameters['required']}, types=[int, int]")


def test_fn_to_tool_with_defaults():
    """Parameters with defaults become optional in the schema."""
    td = fn_to_tool(search_files, tags=["code"])

    assert "pattern" in td.parameters["required"]
    assert "path" not in td.parameters["required"]  # has default
    assert "case_sensitive" not in td.parameters["required"]  # has default
    assert td.parameters["properties"]["path"]["default"] == "."
    assert td.parameters["properties"]["case_sensitive"]["default"] is True
    assert td.tags == ["code"]
    print(f"  ✅ fn_to_tool(search_files): required={td.parameters['required']}")
    print(f"      defaults: path='.', case_sensitive=True")
    print(f"      tags={td.tags}")


def test_openai_schema_format():
    """Generated schema matches OpenAI function-calling format exactly."""
    td = fn_to_tool(add_numbers)
    schema = td.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add_numbers"
    assert schema["function"]["description"] == "Add two numbers together."
    assert schema["function"]["parameters"]["type"] == "object"
    print(f"  ✅ OpenAI schema: {json.dumps(schema, indent=2)[:200]}...")


def test_registry_operations():
    """Register, lookup, filter, and list tools."""
    registry = ToolRegistry()

    td1 = fn_to_tool(add_numbers, tags=["math"])
    td2 = fn_to_tool(search_files, tags=["code", "file"])

    registry.register(td1)
    registry.register(td2)

    assert len(registry) == 2
    assert "add_numbers" in registry
    assert registry.get("add_numbers").name == "add_numbers"
    print(f"  ✅ Registry: {len(registry)} tools registered")

    # Filter by tags
    math_tools = registry.get_tools(tags=["math"])
    assert len(math_tools) == 1 and math_tools[0].name == "add_numbers"
    print(f"  ✅ Filter by tag 'math': {[t.name for t in math_tools]}")

    code_tools = registry.get_tools(tags=["code"])
    assert len(code_tools) == 1 and code_tools[0].name == "search_files"
    print(f"  ✅ Filter by tag 'code': {[t.name for t in code_tools]}")

    # Filter by names
    specific = registry.get_tools(names=["add_numbers"])
    assert len(specific) == 1
    print(f"  ✅ Filter by name: {[t.name for t in specific]}")

    # Get OpenAI schemas
    schemas = registry.get_openai_schemas()
    assert len(schemas) == 2
    assert all(s["type"] == "function" for s in schemas)
    print(f"  ✅ OpenAI schemas: {len(schemas)} tools")


def test_tool_invocation_sync():
    """Invoke a registered tool with JSON arguments."""
    registry = ToolRegistry()
    registry.register(fn_to_tool(add_numbers))

    result = asyncio.get_event_loop().run_until_complete(
        registry.invoke("add_numbers", '{"a": 5, "b": 3}', tool_call_id="tc-1")
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.content == "8"
    assert result.tool_call_id == "tc-1"
    print(f"  ✅ Sync invoke: add_numbers(5, 3) = {result.content}")


def test_tool_invocation_async():
    """Invoke an async tool handler."""
    registry = ToolRegistry()
    registry.register(fn_to_tool(async_fetch, tags=["web"]))

    result = asyncio.get_event_loop().run_until_complete(
        registry.invoke("async_fetch", '{"url": "https://example.com"}', tool_call_id="tc-2")
    )

    assert result.success is True
    assert "example.com" in result.content
    print(f"  ✅ Async invoke: async_fetch('https://example.com') → {result.content[:50]}...")


def test_tool_invocation_error():
    """Invoking a nonexistent tool returns a clean error."""
    registry = ToolRegistry()

    result = asyncio.get_event_loop().run_until_complete(
        registry.invoke("nonexistent", "{}", tool_call_id="tc-3")
    )

    assert result.success is False
    assert "not found" in result.error
    print(f"  ✅ Missing tool: success={result.success}, error='{result.error}'")


def test_tool_invocation_bad_args():
    """Tool with wrong arguments returns a clean error, not a crash."""
    registry = ToolRegistry()
    registry.register(fn_to_tool(add_numbers))

    result = asyncio.get_event_loop().run_until_complete(
        registry.invoke("add_numbers", '{"x": 1}', tool_call_id="tc-4")
    )

    assert result.success is False
    assert result.error is not None
    print(f"  ✅ Bad args: success={result.success}, error='{result.error[:60]}...'")


if __name__ == "__main__":
    print("\n🧪 Test 3: Tool Registry")
    print("-" * 50)
    test_fn_to_tool_basic()
    test_fn_to_tool_with_defaults()
    test_openai_schema_format()
    test_registry_operations()
    test_tool_invocation_sync()
    test_tool_invocation_async()
    test_tool_invocation_error()
    test_tool_invocation_bad_args()
    print("✅ All tool registry tests passed\n")
