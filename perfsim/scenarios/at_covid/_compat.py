"""Compatibility shims for running agent-torch 0.6.0 bundled covid model.

agent_torch 0.6.0's bundled covid action.py imports langchain 0.x APIs that
no longer exist in langchain 1.x. Its `read_config` helper registers
OmegaConf resolvers non-idempotently, so a second call crashes. Its
`config.yaml` hardcodes the AT authors' machine population_dir path.

These workarounds let the bundled covid sim load and re-load (as perfsim's
Simulator.reset rebuilds the runner) inside this Python process. Each is
idempotent; calling install_compat_shims() multiple times is safe.

If agent_torch ships fixes for any of these, the corresponding shim becomes
a no-op without changes here.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_LANGCHAIN_SHIM_INSTALLED = False
_OMEGACONF_RESOLVERS_REGISTERED = False


def install_langchain_shim() -> None:
    """Stub the langchain 0.x symbols that covid's action.py references.

    The real classes are only constructed in LLM mode. Heuristic mode (our
    default; we register our own action) never touches them. Idempotent.
    """
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
    """Return True iff `read_config` should register OmegaConf resolvers.

    Flips to False after the first call so subsequent `read_config` calls
    skip the non-idempotent registration step that otherwise crashes with
    `ValueError: resolver 'sum' is already registered`.
    """
    global _OMEGACONF_RESOLVERS_REGISTERED
    if not _OMEGACONF_RESOLVERS_REGISTERED:
        _OMEGACONF_RESOLVERS_REGISTERED = True
        return True
    return False


def locate_agent_torch_root() -> Path:
    """Find the installed agent_torch package directory.

    Used to resolve the bundled astoria population path without hardcoding
    a machine-specific absolute path.
    """
    import agent_torch

    return Path(agent_torch.__file__).parent


def bundled_astoria_dir() -> Path:
    return locate_agent_torch_root() / "populations" / "astoria"


def bundled_covid_yaml() -> Path:
    return locate_agent_torch_root() / "models" / "covid" / "yamls" / "config.yaml"
