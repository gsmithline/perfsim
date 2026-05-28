"""Compatibility shims for running agent_torch's bundled macro_economics model."""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import agent_torch
    import agent_torch.core.substep
    import agent_torch.core.helpers
    import agent_torch.core.helpers.distributions
    _HAS_AGENT_TORCH = True
except ImportError:
    agent_torch = None  # type: ignore[assignment]
    _HAS_AGENT_TORCH = False

_AGENTTORCH_ALIAS_INSTALLED = False
_OMEGACONF_RESOLVERS_REGISTERED = False


def install_agenttorch_alias() -> None:
    """Register `AgentTorch.*` as an alias for `agent_torch.*`."""
    global _AGENTTORCH_ALIAS_INSTALLED
    if _AGENTTORCH_ALIAS_INSTALLED:
        return
    if not _HAS_AGENT_TORCH:
        raise ImportError("agent_torch is required but not installed")

    sys.modules.setdefault("AgentTorch", agent_torch)
    sys.modules.setdefault("AgentTorch.substep", agent_torch.core.substep)
    sys.modules.setdefault("AgentTorch.helpers", agent_torch.core.helpers)
    sys.modules.setdefault(
        "AgentTorch.helpers.distributions", agent_torch.core.helpers.distributions
    )
    _AGENTTORCH_ALIAS_INSTALLED = True


def should_register_resolvers() -> bool:
    """Return True on first call, False after; prevents duplicate OmegaConf resolver registration."""
    global _OMEGACONF_RESOLVERS_REGISTERED
    if not _OMEGACONF_RESOLVERS_REGISTERED:
        _OMEGACONF_RESOLVERS_REGISTERED = True
        return True
    return False


def locate_agent_torch_root() -> Path:
    """Find the installed agent_torch package directory."""
    if agent_torch is None:
        raise ImportError("agent_torch is required but not installed")
    return Path(agent_torch.__file__).parent


def bundled_nyc_dir() -> Path:
    """Bundled NYC population directory."""
    return locate_agent_torch_root() / "populations" / "NYC"


def subsample_population_dir(n_agents: int, source: Path | None = None) -> Path:
    """Create a temp directory with the bundled NYC population subsampled to first `n_agents` rows."""
    if source is None:
        source = bundled_nyc_dir()

    out_dir = Path(tempfile.mkdtemp(prefix="perfsim-macro-pop-"))
    PER_AGENT = ["age", "area", "county", "ethnicity", "gender", "region"]
    for name in PER_AGENT:
        src_p = source / f"{name}.pickle"
        if not src_p.exists():
            continue
        with open(src_p, "rb") as f:
            series = pickle.load(f)
        subset = series.iloc[:n_agents]
        with open(out_dir / f"{name}.pickle", "wb") as f:
            pickle.dump(subset, f)

    for f in source.iterdir():
        if f.name.endswith(".pickle") and f.stem in PER_AGENT:
            continue
        if f.is_file():
            shutil.copy(f, out_dir / f.name)
    return out_dir


def bundled_macro_yaml(name: str = "config_100_agents.yaml") -> Path:
    """Path to a bundled macro_economics YAML config."""
    return (
        locate_agent_torch_root() / "models" / "macro_economics" / "yamls" / name
    )


def patched_macro_yaml(
    src_yaml: Path | None = None,
    population_dir: Path | None = None,
    n_agents: int | None = None,
) -> str:
    """Rewrite the bundled macro YAML so hardcoded paths resolve correctly."""
    if src_yaml is None:
        src_yaml = bundled_macro_yaml()
    if population_dir is None:
        population_dir = bundled_nyc_dir()

    text = src_yaml.read_text()
    text = text.replace(
        "/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC/100_sampled_agents",
        str(population_dir),
    )
    text = text.replace(
        "/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC",
        str(population_dir),
    )
    if n_agents is not None:
        import re
        text = re.sub(r"(num_agents:\s*)\d+", rf"\g<1>{n_agents}", text)
    if "\n  calibration:" not in text and "calibration:" not in text.split("state:")[0]:
        text = text.replace(
            "simulation_metadata:\n",
            "simulation_metadata:\n  calibration: false\n",
            1,
        )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name
