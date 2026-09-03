"""Every environment variable the frontier launchers set must be one the
runner actually READS.

This exists because of a real near-miss on 2026-08-31. The local launcher
set EPS_SOCIAL=0.2 and POPULATION_UPDATE=anch2; the runner reads neither.
It reads EPS (default 0.3) and selects the operator from
AI_GATE_REFERENCE. It also reads GAMMA_BIAS, whose default is 1.5 --
homophily selection bias, which this project forbids outright -- and the
launcher never set it. A paid run would have produced a
wrong-surface trajectory that looked entirely healthy: right tag, right
model, complete artifacts, wrong dynamics.

A variable that is set but never read is silent, so the test is
mechanical: parse the names out of the launcher, and require each to
appear as a string literal in the runner.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/scripts/cluster_pipelines/run_pokec_gated_lm.py"
SH = REPO / "experiments/scripts/cluster_pipelines/run_frontier_local.sh"
GEN = REPO / "experiments/condor/gen_frontier_icl.py"

# set by the shell/Condor machinery, not consumed by the runner's env parser
EXEMPT = {
    "RUN_TAG", "OUT_DIR", "REPO", "CONDA_SH", "ENV_NAME", "USE_TF",
    "WANDB_MODE", "WANDB_DISABLED", "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_FILE", "MODEL_BACKEND", "PYTHONPATH",
}


def _runner_src() -> str:
    return RUNNER.read_text()


def _names(text: str) -> set[str]:
    # NAME=value pairs on the runner invocation / environment line
    return {m.group(1) for m in
            re.finditer(r"\b([A-Z][A-Z0-9_]{2,})=", text)} - EXEMPT


def test_local_launcher_sets_only_variables_the_runner_reads():
    src = _runner_src()
    body = SH.read_text()
    # only the block that invokes the runner
    start = body.index('RUN_TAG="$TAG"')
    invocation = body[start:body.index("run_pokec_gated_lm.py", start)]
    unread = sorted(n for n in _names(invocation)
                    if f'"{n}"' not in src)
    assert not unread, (
        f"the local launcher sets variables the runner never reads: "
        f"{unread}. A variable that is set but not read is SILENT -- the "
        f"run completes on the wrong surface and looks healthy.")


def test_condor_sub_sets_only_variables_the_runner_reads():
    src = _runner_src()
    gen = GEN.read_text()
    env_line = next(l for l in gen.splitlines()
                    if l.strip().startswith("environment"))
    unread = sorted(n for n in _names(env_line) if f'"{n}"' not in src)
    assert not unread, (
        f"the Condor sub sets variables the runner never reads: {unread}")


def test_gamma_bias_is_pinned_to_zero_everywhere():
    """GAMMA_BIAS defaults to 1.5 in the runner. This project forbids
    homophily selection bias, so it must never be left implicit."""
    for f in (SH, GEN):
        assert "GAMMA_BIAS=0" in f.read_text(), f"{f.name} must pin GAMMA_BIAS=0"


def test_the_section3a_surface_is_pinned():
    body = SH.read_text()
    for knob in ("EPS=0.2", "W_PLAT=1", "INNATE_LAMBDA=1",
                 "DEFFUANT_ALPHA=0.5", "AB_SWEEPS=100",
                 "AI_GATE_REFERENCE=anchor", "AI_GATE_MODE=all_open",
                 "PEER_GATE_MODE=all_open", "ICL_K=0", "ICL_DAYS=8"):
        assert knob in body, f"local launcher must pin {knob}"


# ---- the audit itself must stay honest ---------------------------------
def test_audit_surface_covers_the_knobs_that_caused_the_near_miss():
    """EPS and GAMMA_BIAS are the two the launcher silently got wrong.
    They must be in the compared surface, not merely set somewhere."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_aud", str(REPO / "experiments/scripts/cluster_pipelines/"
                    "audit_frontier_config.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for knob in ("eps", "gamma_bias", "innate_lambda", "w_plat",
                 "ai_gate_mode", "peer_gate_mode", "population_update",
                 "ai_gate_reference", "ab_sweeps", "icl_k", "icl_days"):
        assert knob in m.SURFACE, f"{knob} must be compared field-by-field"
    # and the two must be distinguished, not conflated
    assert "innate_lambda" in m.SURFACE and "gamma_bias" in m.SURFACE


def test_no_launcher_sets_the_retired_population_update():
    """POPULATION_UPDATE is not read by the runner. Leaving it set is
    misleading: it looks like the operator is pinned when it is not."""
    for f in (SH, GEN):
        assert "POPULATION_UPDATE=" not in f.read_text(), (
            f"{f.name} still sets POPULATION_UPDATE, which the runner "
            f"ignores; the operator comes from AI_GATE_REFERENCE")


def test_explicit_answer_limit_is_pinned_not_none():
    body = SH.read_text()
    assert 'OR_MAX_TOKENS="$OR_MAX_TOKENS"' in body
    pre = (REPO / "experiments/scripts/cluster_pipelines/or_preflight.py"
           ).read_text()
    assert "ANSWER_TOKENS = 32" in pre
    assert 'OR_ANSWER_TOKEN_LIMIT' in pre
