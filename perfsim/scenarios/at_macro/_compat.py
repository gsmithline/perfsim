"""Compatibility shims for running agent_torch's bundled macro_economics model.

The macro_economics substeps were written against an older `AgentTorch.*`
(capital-A) namespace that the installed agent_torch 0.6.0 package does NOT
expose. Modules like `AgentTorch.substep`, `AgentTorch.helpers`,
`AgentTorch.helpers.distributions` are unresolvable on import.

Additionally the bundled action.py imports `langchain` 0.x APIs (same
issue as covid) and a `macro_economics.prompt` relative import that breaks.

These shims map the capital-A namespace onto the installed lowercase
package so the bundled transition substeps can load. The bundled action
substep is NEVER imported because perfsim replaces it with
`PerfsimEarningDecision` (action.py), so we don't need to fix the
langchain or macro_economics.prompt imports here.

Each helper is idempotent; calling install_compat_shims() multiple times
is safe.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_AGENTTORCH_ALIAS_INSTALLED = False
_OMEGACONF_RESOLVERS_REGISTERED = False


def install_agenttorch_alias() -> None:
    """Register `AgentTorch.*` as an alias for `agent_torch.*`.

    The bundled macro_economics substep files do
    `from AgentTorch.substep import SubstepTransition`. agent_torch 0.6.0
    exposes the same class at `agent_torch.core.substep.SubstepTransition`.
    We populate sys.modules to make both paths resolve to the same module.
    Idempotent.
    """
    global _AGENTTORCH_ALIAS_INSTALLED
    if _AGENTTORCH_ALIAS_INSTALLED:
        return

    import agent_torch
    import agent_torch.core.substep
    import agent_torch.core.helpers
    import agent_torch.core.helpers.distributions

    sys.modules.setdefault("AgentTorch", agent_torch)
    sys.modules.setdefault("AgentTorch.substep", agent_torch.core.substep)
    sys.modules.setdefault("AgentTorch.helpers", agent_torch.core.helpers)
    sys.modules.setdefault(
        "AgentTorch.helpers.distributions", agent_torch.core.helpers.distributions
    )
    _AGENTTORCH_ALIAS_INSTALLED = True


def should_register_resolvers() -> bool:
    """Return True iff `read_config` should register OmegaConf resolvers.

    Same pattern as at_covid: flips to False after the first call so
    subsequent `read_config` calls skip the non-idempotent registration
    step that otherwise crashes with `ValueError: resolver 'sum' is
    already registered`.
    """
    global _OMEGACONF_RESOLVERS_REGISTERED
    if not _OMEGACONF_RESOLVERS_REGISTERED:
        _OMEGACONF_RESOLVERS_REGISTERED = True
        return True
    return False


def locate_agent_torch_root() -> Path:
    """Find the installed agent_torch package directory."""
    import agent_torch

    return Path(agent_torch.__file__).parent


def bundled_nyc_dir() -> Path:
    """Bundled NYC population directory. Contains age/area/gender/etc. pickles
    plus county-level COVID + unemployment time series referenced by the
    macro_economics substeps.
    """
    return locate_agent_torch_root() / "populations" / "NYC"


def subsample_population_dir(n_agents: int, source: Path | None = None) -> Path:
    """Create a temp directory with the bundled NYC population subsampled
    to first `n_agents` rows.

    Why: the bundled NYC pickles contain ~2.7M agents (full population).
    The bundled YAML's `population_dir: .../100_sampled_agents` references
    a 100-agent subdirectory that AT did NOT ship. Pointing at the full
    NYC dir causes shape mismatches (some state tensors init at
    num_agents=100, others load 2.7M from pickles).

    This helper:
      1. Reads each per-agent pickle (pandas Series) from `source` (or
         bundled NYC dir if None)
      2. Slices to first `n_agents` rows
      3. Pickles back to a fresh temp directory
      4. Also copies mapping.json + any other non-per-agent files

    Returns the temp dir path. n_agents must match the YAML's num_agents.
    """
    import pickle
    import shutil
    import tempfile

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

    # Copy everything else (mapping.json, CSVs, household, etc.) verbatim.
    # The non-per-agent files (CSVs of unemployment / COVID cases) are
    # time-series, not per-agent, so they should NOT be subsampled.
    for f in source.iterdir():
        if f.name.endswith(".pickle") and f.stem in PER_AGENT:
            continue
        if f.is_file():
            shutil.copy(f, out_dir / f.name)
    return out_dir


def bundled_macro_yaml(name: str = "config_100_agents.yaml") -> Path:
    """Path to a bundled macro_economics YAML config.

    Default is `config_100_agents.yaml`, the smallest config. Other options:
    `config.yaml` (full), `config_nyc_100_agents.yaml` (NYC-specific).
    """
    return (
        locate_agent_torch_root() / "models" / "macro_economics" / "yamls" / name
    )


def patched_macro_yaml(
    src_yaml: Path | None = None,
    population_dir: Path | None = None,
) -> str:
    """Rewrite the bundled macro YAML so all paths resolve under the
    installed agent_torch package, not the AT authors' machine.

    The bundled YAML has `population_dir: /Users/shashankkumar/...`. We
    replace with the bundled NYC dir (or a user-supplied path) and write to
    a temp file. Returns the temp file's path as a string.
    """
    if src_yaml is None:
        src_yaml = bundled_macro_yaml()
    if population_dir is None:
        population_dir = bundled_nyc_dir()

    text = src_yaml.read_text()
    text = text.replace(
        "/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC/100_sampled_agents",
        str(population_dir),
    )
    # Older bundled YAMLs sometimes also reference the parent NYC dir directly.
    text = text.replace(
        "/Users/shashankkumar/Documents/GitHub/MacroEcon/populations/NYC",
        str(population_dir),
    )
    # The bundled config_100_agents.yaml lacks a `calibration` field that
    # AT's SubstepAction.__init__ now reads. Inject a default `False` under
    # simulation_metadata if not present so substeps can construct.
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
