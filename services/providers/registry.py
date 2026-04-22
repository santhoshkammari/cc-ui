"""
Provider registry — maps model names to provider classes.
Single source of truth for which providers are available.
"""
from __future__ import annotations

import logging
from typing import Type

from .base import BaseProvider, ProviderConfig

log = logging.getLogger("cc-ui.registry")

_PROVIDERS: dict[str, Type[BaseProvider]] = {}
_ALIASES: dict[str, str] = {}


def register(name: str, provider_cls: Type[BaseProvider], aliases: list[str] | None = None):
    """Register a provider class under a canonical name."""
    _PROVIDERS[name] = provider_cls
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def get_provider(name: str) -> BaseProvider:
    """Instantiate a provider by name or alias."""
    canonical = _ALIASES.get(name, name)
    cls = _PROVIDERS.get(canonical)
    if cls is None:
        available = list(_PROVIDERS.keys()) + list(_ALIASES.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return cls()


def list_providers() -> list[dict]:
    """Return capabilities of all registered providers."""
    seen = set()
    result = []
    for name, cls in _PROVIDERS.items():
        if name not in seen:
            seen.add(name)
            instance = cls()
            info = instance.get_capabilities()
            info["aliases"] = [a for a, c in _ALIASES.items() if c == name]
            result.append(info)
    return result


async def health_check_all() -> dict:
    """Run health checks on all providers."""
    results = {}
    for name, cls in _PROVIDERS.items():
        try:
            instance = cls()
            results[name] = await instance.health_check()
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
    return results


def _auto_register():
    """Auto-register all built-in providers."""
    try:
        from .claude import ClaudeProvider
        register("claude", ClaudeProvider, aliases=["claude-code", "cc"])
    except ImportError as e:
        log.debug("Claude provider not available: %s", e)

    try:
        from .opencode import OpenCodeProvider
        register("opencode", OpenCodeProvider, aliases=["oc"])
    except ImportError as e:
        log.debug("OpenCode provider not available: %s", e)

    try:
        from .qwen import QwenProvider
        register("qwen", QwenProvider, aliases=["qwen-code"])
    except ImportError as e:
        log.debug("Qwen provider not available: %s", e)

    try:
        from .vllm import VLLMProvider
        register("vllm", VLLMProvider, aliases=["vllm-local"])
    except ImportError as e:
        log.debug("vLLM provider not available: %s", e)

    try:
        from .gemini import GeminiProvider
        register("gemini", GeminiProvider, aliases=["google", "gemini-pro"])
    except ImportError as e:
        log.debug("Gemini provider not available: %s", e)

    try:
        from .copilot import CopilotProvider
        register("copilot", CopilotProvider, aliases=["gh-copilot"])
    except ImportError as e:
        log.debug("Copilot provider not available: %s", e)

    try:
        from .kivi import KiviProvider
        register("kivi", KiviProvider, aliases=["kivi-local"])
    except ImportError as e:
        log.debug("Kivi provider not available: %s", e)

    try:
        from .openai_agent import OpenAIAgentProvider
        register("openai-agent", OpenAIAgentProvider, aliases=["oai-agent", "openai"])
    except ImportError as e:
        log.debug("OpenAI Agent provider not available: %s", e)

    try:
        from .inhouse import InhouseProvider
        register("inhouse", InhouseProvider, aliases=["ai-framework", "lab"])
    except ImportError as e:
        log.debug("Inhouse provider not available: %s", e)

    try:
        from .orchestrator_provider import OrchestratorProvider
        register("claudeagents", OrchestratorProvider, aliases=["multi-agent", "orchestrator"])
    except ImportError as e:
        log.debug("Orchestrator provider not available: %s", e)


_auto_register()
