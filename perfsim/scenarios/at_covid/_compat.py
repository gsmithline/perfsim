"""Compatibility shims for running agent-torch 0.6.0 bundled covid model."""

from __future__ import annotations

import sys
import types
from pathlib import Path

try:
    import agent_torch as _agent_torch
    _HAS_AGENT_TORCH = True
except ImportError:
    _agent_torch = None  # type: ignore[assignment]
    _HAS_AGENT_TORCH = False

_LANGCHAIN_SHIM_INSTALLED = False
_OMEGACONF_RESOLVERS_REGISTERED = False


def install_langchain_shim() -> None:
    """Stub the langchain 0.x symbols that covid's action.py references."""
    global _LANGCHAIN_SHIM_INSTALLED
    if _LANGCHAIN_SHIM_INSTALLED:
        return
    chains = types.ModuleType("langchain.chains")
    chains.LLMChain = type("LLMChain", (), {"__init__": lambda *a, **kw: None})
    sys.modules["langchain.chains"] = chains

    prompts = types.ModuleType("langchain.prompts")
    for name in (
        "ChatPromptTemplate",
        "HumanMessagePromptTemplate",
        "SystemMessagePromptTemplate",
        "MessagesPlaceholder",
    ):
        setattr(prompts, name, type(name, (), {"__init__": lambda *a, **kw: None}))
    sys.modules["langchain.prompts"] = prompts

    _LANGCHAIN_SHIM_INSTALLED = True


def should_register_resolvers() -> bool:
    """Return True on first call, False after; prevents duplicate OmegaConf resolver registration."""
    global _OMEGACONF_RESOLVERS_REGISTERED
    if not _OMEGACONF_RESOLVERS_REGISTERED:
        _OMEGACONF_RESOLVERS_REGISTERED = True
        return True
    return False


def locate_agent_torch_root() -> Path:
    """Find the installed agent_torch package directory."""
    if _agent_torch is None:
        raise ImportError("agent_torch is required but not installed")
    return Path(_agent_torch.__file__).parent


def bundled_astoria_dir() -> Path:
    return locate_agent_torch_root() / "populations" / "astoria"


def bundled_covid_yaml() -> Path:
    return locate_agent_torch_root() / "models" / "covid" / "yamls" / "config.yaml"
