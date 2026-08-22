"""ADVERSARIAL tests for analyze_section3.py.

The analyzer is where a technically-correct wave turns into a wrong
claim. The failures it can produce are all of the "renders beautifully"
kind:

  * a population series that starts at round 1 instead of the innate
    vector, so every panel silently omits the only point that shows what
    the prior WAS;
  * a within-round intermediate state (post-AI, pre-peer) leaking into a
    series that is documented as post-peer, which makes the AI step look
    like a population movement;
  * a late window that is off by one, or by ten;
  * a drifting cell reported as an equilibrium, which is the difference
    between "the prior is retained" and "the prior has not finished
    disappearing yet";
  * a missing arm quietly dropped from the grid, so the figure has a
    hole where the strongest counter-evidence would have been.

CONTRACT
  101 population points per cell: t = 0 is the INNATE vector, t = 1..100
  are the END-OF-ROUND POST-PEER states (op_raw[0..99]).
  Late window = rounds 81..100 -> op_raw indices 80..99.
  A missing arm is a HARD FAILURE, never a hole in the grid.
"""
from __future__ import annotations

import gzip
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import zlib

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines")

N_AGENTS = 723
ROUNDS = 100
N_POINTS = 101                 # innate + 100 post-peer states
LATE_LO, LATE_HI = 81, 100     # 1-indexed rounds; op_raw[80:100]
H100 = "NVIDIA H100 80GB HBM3"
QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen3-8B"
POP_UPDATE = "nested_ai_anchored_then_social_v2"

MODELS = {"qwen7b": QWEN25, "qwen3_8b": QWEN3}
FWD_LAMS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
REV_LAMS = [1.0, 8.0]
ENVS = [(0.5, 1.0), (1.0, 1.0), (0.5, 0.2)]
REV_ENVS = [(0.5, 1.0), (1.0, 1.0)]
REUSED = {
    ("qwen7b", "sft", (0.5, 1.0)):
        "pofdqwu_qwen7b_b0_eaopen_w0p5_l1_esopen_s0_r100",
    ("qwen7b", "sft", (1.0, 1.0)):
        "pofdqwu_qwen7b_b0_eaopen_w1_l1_esopen_s0_r100",
    ("qwen7b", "fwdlam1", (0.5, 1.0)):
        "pofdqwu_qwen7b_b1_eaopen_w0p5_l1_esopen_s0_r100",
    ("qwen7b", "fwdlam1", (1.0, 1.0)):
        "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r100",
}


def _num(v):
    return f"{v:g}".replace(".", "p")


def _seed_of(tag):
    """Stable across processes -- unlike hash(), which is randomized."""
    return zlib.crc32(tag.encode()) % 10000


def _find_analyzer():
    for c in (os.path.join(PIPE, "analyze_section3.py"),
              os.path.join(ROOT, "experiments", "condor",
                           "analyze_section3.py"),
              os.path.join(ROOT, "analyze_section3.py")):
        if os.path.exists(c):
            return c
    return None


ANALYZE = _find_analyzer()
needs_analyzer = pytest.mark.skipif(
    ANALYZE is None, reason="analyze_section3.py has not landed yet")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_analyze_s3", ANALYZE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_analyze_s3"] = m
    spec.loader.exec_module(m)
    return m


def _pick(m, names, contains=None):
    """A module-level callable, by exact name then by substring."""
    for n in names:
        f = getattr(m, n, None)
        if callable(f):
            return f
    if contains:
        cands = [v for k, v in vars(m).items()
                 if callable(v) and not k.startswith("__")
                 and all(c in k.lower() for c in contains)
                 and getattr(v, "__module__", None) == m.__name__]
        if len(cands) == 1:
            return cands[0]
    return None


def _require(fn, names, what):
    if fn is None:
        pytest.fail(
            f"analyze_section3.py exposes no module-level {what} helper. "
            f"Looked for {names}. This wave's numbers are not reviewable "
            f"unless the metric functions are importable and testable in "
            f"isolation -- every other analyzer in "
            f"experiments/scripts/cluster_pipelines/ exposes one.")
    return fn


# ------------------------------------------------------------------ fixtures
def _innate(n=N_AGENTS, seed=1234):
    return torch.rand(n, generator=torch.Generator().manual_seed(seed))


def _stationary_op(seed=0, n=N_AGENTS, rounds=ROUNDS):
    """A cell that has settled: small zero-mean jitter about a fixed
    level from round ~30 on."""
    g = torch.Generator().manual_seed(seed)
    base = 0.30 + 0.20 * torch.rand(n, generator=g)
    op = base.unsqueeze(0).repeat(rounds, 1)
    decay = torch.tensor([math.exp(-t / 8.0) for t in range(rounds)])
    op = op + decay.unsqueeze(1) * 0.15
    op = op + 0.002 * torch.randn(rounds, n, generator=g)
    return op.clamp(0.0, 1.0)


def _drifting_op(seed=0, n=N_AGENTS, rounds=ROUNDS):
    """A cell that is STILL MOVING at round 100: a steady ramp that does
    not flatten. Its round-100 state is a late-round state, not an
    equilibrium, and reporting it as one would be the wave's most
    expensive mistake."""
    g = torch.Generator().manual_seed(seed)
    base = 0.10 + 0.05 * torch.rand(n, generator=g)
    ramp = torch.linspace(0.0, 0.70, rounds).unsqueeze(1)
    return (base.unsqueeze(0) + ramp
            + 0.002 * torch.randn(rounds, n, generator=g)).clamp(0.0, 1.0)


def _mk_run(root, tag, *, model="qwen7b", style="sft_kl", kl_beta=1.0,
            kl_direction="forward", w_plat=0.5, innate_lambda=1.0,
            rounds=ROUNDS, n=N_AGENTS, op=None, pred=None, innate=None):
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    base = MODELS["qwen3_8b" if model == "qwen3_8b" else "qwen7b"]
    cfg = {
        "base_model": base, "seed": 0, "seed_base_data": 1,
        "dataset": "movielens", "ml_target": "Action", "n_labeled": n,
        "training_style": style, "kl_beta": kl_beta,
        "kl_direction": kl_direction, "kl_ref_adapter": "",
        "anchor_mode": "fixed", "w_plat": w_plat,
        "innate_lambda": innate_lambda, "ai_gate_mode": "all_open",
        "peer_gate_mode": "all_open", "eps": 0.2, "eps_ai": 1.0,
        "ab_sweeps": 1, "pop_model": "ab", "fresh_each_round": True,
        "use_lora": True, "lora_r": 512, "sft_lr": 5e-05, "sft_epochs": 1,
        "epoch_size": 100, "max_steps": None, "train_cap": 723,
        "icl_k": 0, "icl_days": 0, "n_rounds": rounds,
        "serve_eval_mode": True, "population_update": POP_UPDATE,
        "chat_thinking": 0 if model == "qwen3_8b" else "default",
        "hardware": {"gpu_name": H100},
    }
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    if innate is None:
        innate = _innate(n)
    if op is None:
        op = _stationary_op(seed=_seed_of(tag), n=n, rounds=rounds)
    if pred is None:
        g = torch.Generator().manual_seed(3)
        pred = 0.2 + 0.6 * torch.rand(rounds, n, generator=g)
    torch.save({"trajectory": [{"round": t} for t in range(rounds)],
                "config": cfg, "op_raw": op, "pred_raw": pred,
                "innate": innate,
                "profiles": [{"i": i} for i in range(n)],
                "twin_raw": op.clone(),
                "probe_idx": list(range(32))},
               os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(rounds):
            fh.write(json.dumps({
                "round": t, "l_init": 2.0 / (t + 1), "grad_norm0": 3.0,
                "grad_kl_norm0": 0.0 if t == 0 else 1.5,
                "grad_cos0": 0.1, "grad_ratio0": 0.3,
                "n_train": 100}) + "\n")
    with gzip.open(os.path.join(d, "raw_gen_log.json.gz"), "wt") as fh:
        for t in range(rounds):
            fh.write(json.dumps({"round": t, "parse_fail_frac": 0.0,
                                 "raw": ["0.42"], "parsed": [0.42]}) + "\n")
    return d


def _cells():
    """All 50 conceptual (model, arm, env, kl_beta, kldir, style) cells."""
    out = []
    for m in MODELS:
        for env in ENVS:
            out.append((m, "sft", env, 0.0, "forward", "sft"))
            for lam in FWD_LAMS:
                out.append((m, f"fwdlam{_num(lam)}", env, lam, "forward",
                            "sft_kl"))
        for env in REV_ENVS:
            for lam in REV_LAMS:
                out.append((m, f"revlam{_num(lam)}", env, lam, "reverse",
                            "sft_kl"))
    return out


DRIFT_TAG = "pofds3_qwen3_8b_fwdlam0p1_eaopen_w0p5_k1_esopen_anch2_s0_r100"


def _build_grid(root, *, drop=None, drift=DRIFT_TAG):
    """The whole wave on disk: 46 pofds3_ cells + the 4 archived QWU
    cells they reuse. `drop` removes one tag entirely."""
    innate = _innate()
    tags = {}
    for m, arm, env, lam, kldir, style in _cells():
        w, k = env
        tag = REUSED.get((m, arm, env))
        if tag is None:
            tag = (f"pofds3_{m}_{arm}_eaopen_w{_num(w)}_k{_num(k)}"
                   f"_esopen_anch2_s0_r100")
        if tag == drop:
            continue
        op = (_drifting_op(seed=1) if tag == drift
              else _stationary_op(seed=_seed_of(tag)))
        _mk_run(root, tag, model=m, style=style, kl_beta=lam,
                kl_direction=kldir, w_plat=w, innate_lambda=k,
                op=op, innate=innate)
        tags[(m, arm, env)] = tag
    _write_reuse_manifest(root, tags)
    return tags


def _sha_t(t):
    import hashlib
    a = torch.as_tensor(t).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _write_reuse_manifest(root, tags):
    """Bind the four archived pofdqwu_ dirs to the Section-3 slots they
    satisfy. Without it the analyzer cannot know that a non-grammar tag
    stands in for a conceptual cell, and would report those four slots
    absent -- which would make the missing-arm test below indiscriminate.
    """
    cells = []
    for (m, arm, env), tag in tags.items():
        if not tag.startswith("pofdqwu_"):
            continue
        d = os.path.join(root, tag)
        blob = torch.load(os.path.join(d, "trajectory.pt"),
                          map_location="cpu", weights_only=False)
        cells.append({"model": m, "arm": arm, "beta": env[0], "k": env[1],
                      "status": "reused", "run_tag": tag, "run_dir": d,
                      "pred_raw_sha256": _sha_t(blob["pred_raw"]),
                      "op_raw_sha256": _sha_t(blob["op_raw"])})
    p = os.path.join(root, "_reuse_manifest.json")
    with open(p, "w") as fh:
        json.dump({"key": "section3", "cells": cells}, fh)
    return p


def _manifest_of(root):
    return os.path.join(root, "_reuse_manifest.json")


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("s3_runs"))
    tags = _build_grid(root)
    return root, tags


# ------------------------------------------------------------------- CLI
def _help():
    r = subprocess.run([sys.executable, ANALYZE, "--help"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "USE_TF": "0"})
    return (r.stdout or "") + (r.stderr or "")


def _root_flag():
    h = _help()
    for f in ("--roots", "--runs-root", "--runs", "--root", "--run-dir",
              "--runs-dir"):
        if f in h:
            return f
    return None


def _out_flag():
    h = _help()
    for f in ("--out", "--out-dir", "--outdir", "--figdir"):
        if f in h:
            return f
    return None


def _run_analyzer(runs_root, out_dir, *extra):
    rf, of = _root_flag(), _out_flag()
    if rf is None:
        pytest.skip("analyze_section3.py exposes no runs-root flag; "
                    "cannot drive it end to end")
    cmd = [sys.executable, ANALYZE, rf, runs_root]
    if of:
        cmd += [of, out_dir]
    mf = _manifest_of(runs_root)
    if os.path.exists(mf) and "--reuse-manifest" in _help():
        cmd += ["--reuse-manifest", mf]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "USE_TF": "0"})


def _blob(r, out_dir):
    """stdout + stderr + every text artifact the run emitted."""
    b = _text(r)
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".csv", ".json", ".txt", ".md")):
                with open(os.path.join(out_dir, f), errors="ignore") as fh:
                    b += "\n" + fh.read()
    return b


def _text(r):
    return (r.stdout or "") + (r.stderr or "")


# ------------------------------------------------------- the W1 helper
@needs_analyzer
def test_w1_helper_matches_a_hand_computed_case(mod):
    """W1 between two equal-size empirical samples is the mean absolute
    difference of the SORTED vectors. Getting this wrong -- e.g.
    differencing unsorted, or dividing by the wrong n -- produces a
    number that is monotone in the right direction and wrong in
    magnitude, which is the hardest kind of error to notice."""
    w1 = _require(_pick(mod, ("w1", "w1_sorted", "w1_quantile",
                             "wasserstein1", "quantile_w1"),
                        contains=("w1",)),
                  "('w1', 'w1_sorted', 'w1_quantile', ...)", "W1")
    a = [0.0, 0.0, 1.0, 1.0]
    b = [1.0, 0.0, 2.0, 1.0]          # deliberately UNSORTED
    # sorted: [0,0,1,1] vs [0,1,1,2] -> |d| = [0,1,0,1] -> 0.5
    assert abs(float(w1(a, b)) - 0.5) < 1e-9, float(w1(a, b))
    # a pure shift: W1 == the shift, exactly
    x = torch.rand(500, generator=torch.Generator().manual_seed(2))
    assert abs(float(w1(x, x + 0.3)) - 0.3) < 1e-5
    # identical samples -> 0, and the metric is symmetric
    assert abs(float(w1(x, x))) < 1e-9
    assert abs(float(w1(x, x + 0.3)) - float(w1(x + 0.3, x))) < 1e-9
    # order of the inputs must not matter (it would if unsorted)
    p = torch.randperm(500, generator=torch.Generator().manual_seed(4))
    assert abs(float(w1(x, x[p] + 0.3)) - 0.3) < 1e-5


# ------------------------------------------- the collapse / mode helpers
def _stats_fn(m):
    """A single dict-returning served-map stats helper, if there is one.

    The five collapse readouts may be exposed individually or as one
    dict; both are fine, the point is that they are computable and
    checkable outside the plotting code."""
    return _pick(m, ("served_stats", "served_map_stats", "mode_stats",
                     "collapse_stats", "degeneracy_stats"),
                 contains=("served", "stats"))


def _handmade():
    """six 0.25, three 0.65, one 0.9 -> distinct 3, largest share 0.6,
    top-3 share 1.0, effective modes exp(H) = 2.45454..."""
    v = torch.tensor([0.25] * 6 + [0.65] * 3 + [0.9])
    p = [0.6, 0.3, 0.1]
    H = -sum(q * math.log(q) for q in p)
    return v, 3, 0.6, 1.0, math.exp(H)


@needs_analyzer
def test_effective_modes_helper_matches_a_hand_computed_case(mod):
    """exp(entropy). Note this is invariant to the log base -- 2 ** H2 ==
    exp(Hn) -- so the only ways to get it wrong are using the wrong
    probabilities or forgetting the exponential."""
    v, _, _, _, want = _handmade()
    stats = _stats_fn(mod)
    if stats is not None:
        d = stats(v)
        key = next((k for k in ("eff_modes", "effective_modes",
                                "eff_support") if k in d), None)
        assert key, f"served-map stats carry no effective-modes field: {d}"
        assert abs(float(d[key]) - want) < 1e-3, f"{d[key]} != {want}"
        return
    fn = _require(_pick(mod, ("eff_modes", "effective_modes", "eff_support",
                             "n_eff_modes", "effective_n_modes"),
                        contains=("eff",)),
                  "('eff_modes', 'effective_modes', 'eff_support', ...)",
                  "effective-modes")
    got = float(fn(v))
    assert abs(got - want) < 1e-3, f"{got} != {want}"


@needs_analyzer
def test_effective_modes_reads_a_coarse_map_as_two_modes(mod):
    """The Qwen2.5 shape: two values to 98.9% of agents. A 50-bin
    histogram would report ~2 as well here, but only because the two
    values happen to fall in different bins -- the EXACT-value entropy
    is the definition that survives a served map whose levels are
    0.6499 and 0.6501."""
    stats = _stats_fn(mod)
    if stats is None:
        pytest.skip("no dict-returning served-map stats helper")
    n = 1000
    v = torch.cat([torch.full((494,), 0.25), torch.full((495,), 0.65),
                   torch.linspace(0.01, 0.99, 11)])
    assert v.numel() == n
    d = stats(v)
    key = next((k for k in ("eff_modes", "effective_modes", "eff_support")
                if k in d), None)
    assert key, d
    assert 2.0 <= float(d[key]) <= 2.6, (
        f"a map serving two values to 98.9% of agents reads as "
        f"{d[key]} effective modes")


@needs_analyzer
def test_mode_share_helpers_match_a_hand_computed_case(mod):
    v, n_distinct, share1, share3, _ = _handmade()
    stats = _stats_fn(mod)
    if stats is not None:
        d = stats(v)
        assert int(d["n_distinct"]) == n_distinct, d
        assert abs(float(d["mode_share"]) - share1) < 1e-9, d
        k3 = next((k for k in ("top3_share", "top3_mode_share",
                               "top_3_share") if k in d), None)
        assert k3, f"served-map stats carry no top-3 share: {d}"
        assert abs(float(d[k3]) - share3) < 1e-9, d
        ksd = next((k for k in ("sd", "pred_sd", "std") if k in d), None)
        assert ksd, f"served-map stats carry no SD: {d}"
        assert abs(float(d[ksd]) - float(v.std(unbiased=False))) < 1e-6, d
        return
    ms = _require(_pick(mod, ("mode_share", "max_mode_share", "mode_mass",
                             "largest_mode_share"),
                        contains=("mode", "share")),
                  "('mode_share', 'max_mode_share', 'mode_mass', ...)",
                  "largest-mode-share")
    assert abs(float(ms(v)) - share1) < 1e-9, float(ms(v))
    top3 = _pick(mod, ("top3_mode_share", "top_3_mode_share",
                       "topk_mode_share", "top3_share"),
                 contains=("top", "share"))
    if top3 is not None:
        try:
            got = float(top3(v))
        except TypeError:
            got = float(top3(v, 3))
        assert abs(got - share3) < 1e-9, got
    nd = _pick(mod, ("n_distinct", "distinct_values", "n_uniq",
                     "n_unique"), contains=("distinct",))
    if nd is not None:
        assert int(nd(v)) == n_distinct


# ------------------------------------------- 101 points, innate at t = 0
def _series_fn(mod):
    return _pick(mod, ("population_series", "pop_series", "series",
                       "with_innate", "prepend_innate", "arm_series",
                       "trajectory_points"), contains=("series",))


@needs_analyzer
def test_population_series_is_101_points_starting_at_the_innate_vector(mod):
    """t = 0 IS the innate vector. Dropping it removes the only point in
    the whole figure that shows what the prior was before the loop
    touched it -- and every "distance travelled" number then measures
    from round 1, understating the movement by exactly the first,
    largest step."""
    fn = _series_fn(mod)
    if fn is None:
        pytest.skip("no population-series helper to test directly; the "
                    "end-to-end variant below covers the same claim")
    innate = _innate()
    op = _stationary_op(seed=1)
    try:
        s = fn(op, innate)
    except TypeError:
        s = fn(innate=innate, op=op)
    n = s.shape[0] if torch.is_tensor(s) else len(s)
    assert n == N_POINTS, (
        f"{n} population points, contract says {N_POINTS} "
        f"(t=0 innate + 100 post-peer)")
    first = s[0]
    got = (float(torch.as_tensor(first).float().mean())
           if not isinstance(first, (int, float)) else float(first))
    want = float(innate.mean())
    assert abs(got - want) < 1e-5, (
        f"series[0] = {got}, innate mean = {want}: t=0 is not the innate "
        f"vector")


@needs_analyzer
def test_late_window_constant_is_rounds_81_to_100(mod):
    """Off by one here moves 1/20th of the window. Off by ten silently
    turns a 20-round average into a 30-round one that includes rounds
    the wave does not claim are settled."""
    names = [k for k in vars(mod) if "late" in k.lower()]
    assert names, ("analyze_section3.py defines no LATE_* constant; the "
                   "late window (rounds 81-100) must be a named constant, "
                   "not an inline slice")
    vals = {k: getattr(mod, k) for k in names}
    flat = []
    for v in vals.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            flat.append(v)
        elif isinstance(v, (list, tuple, range)):
            flat += [x for x in v
                     if isinstance(x, int) and not isinstance(x, bool)]
    assert flat, (
        f"no integer late-window bound among {sorted(vals)}; the window "
        f"must be a named integer constant")
    # accepted spellings: (81, 100) 1-indexed, or (80, 100)/(80, 99) as
    # python slice bounds over op_raw.
    assert min(flat) in (80, 81), (
        f"late window starts at {min(flat)}; contract says round "
        f"{LATE_LO} (op_raw index {LATE_LO - 1}). {vals}")
    assert max(flat) in (99, 100), (
        f"late window ends at {max(flat)}; contract says round {LATE_HI}. "
        f"{vals}")


@needs_analyzer
def test_late_window_selects_exactly_twenty_rounds(mod):
    fn = _pick(mod, ("late_window", "late_slice", "late", "late_rounds"),
               contains=("late",))
    if fn is None or not callable(fn):
        pytest.skip("no late-window helper exposed")
    op = _stationary_op(seed=2)
    sel = fn(op)
    n = sel.shape[0] if torch.is_tensor(sel) else len(sel)
    assert n == 20, f"late window selects {n} rounds, contract says 20"
    if torch.is_tensor(sel):
        assert torch.allclose(sel, op[LATE_LO - 1:LATE_HI]), (
            "late window is not op_raw[80:100]")


# ----------------------------------------- no within-round intermediate state
@needs_analyzer
def test_no_within_round_intermediate_state_is_emitted(mod, grid):
    """op_raw[t] is the END-OF-ROUND POST-PEER state. The runner also
    computes a post-AI, pre-peer state inside the round. If that leaks
    into the series, the AI step reads as a population movement and the
    peer step reads as a correction -- a completely different mechanism
    story, from the same run."""
    root, _ = grid
    banned = ("post_ai", "postai", "pre_peer", "prepeer", "x_ai",
              "mid_round", "intra_round", "op_ai", "half_step")
    src = inspect.getsource(mod).lower()
    hits = [b for b in banned if b in src]
    assert not hits, (
        f"analyzer references within-round intermediate state {hits}; the "
        f"series must be built from op_raw (end-of-round post-peer) only")
    out = os.path.join(root, "_out_noleak")
    os.makedirs(out, exist_ok=True)
    # --allow-missing: the CPU endpoint artifacts (perfect-prediction and
    # frozen-replay .pt files) are a separate prerequisite this synthetic
    # tree does not fabricate; the claim under test is what the emitted
    # CSVs contain.
    r = _run_analyzer(root, out, "--allow-missing")
    if not any(f.endswith(".csv") for f in os.listdir(out)):
        pytest.skip(f"analyzer emitted no CSV to inspect:\n"
                    f"{_text(r)[-1500:]}")
    for f in os.listdir(out):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(out, f)) as fh:
            head = fh.readline().lower()
        assert not any(b in head for b in banned), (out, f, head)


# --------------------------------------------------------- convergence flag
@needs_analyzer
def test_a_drifting_cell_is_flagged_and_not_called_an_equilibrium(grid):
    """A cell still ramping at round 100 has a round-100 STATE, not an
    equilibrium. Calling it one converts "the prior has not finished
    disappearing" into "the prior is retained at this level"."""
    root, _ = grid
    out = os.path.join(root, "_out_conv")
    os.makedirs(out, exist_ok=True)
    r = _run_analyzer(root, out, "--allow-missing")
    blob = _blob(r, out)
    if DRIFT_TAG not in blob:
        pytest.skip(f"analyzer emitted nothing naming the drifting cell:\n"
                    f"{_text(r)[-1500:]}")
    low = blob.lower()
    assert DRIFT_TAG in blob or DRIFT_TAG.replace("pofds3_", "") in blob, (
        f"the analyzer never mentions the drifting cell {DRIFT_TAG}")
    markers = ("not converged", "non-stationary", "nonstationary",
               "not stationary", "still moving", "drift", "unsettled",
               "late-round state", "not an equilibrium", "no equilibrium")
    line = [l for l in blob.splitlines() if DRIFT_TAG in l]
    hit = any(m in low for m in markers)
    assert hit, (
        "a cell that is still ramping at round 100 is reported with no "
        "non-convergence marker anywhere in the output. Searched for "
        f"{markers}.\nlines mentioning the cell: {line[:5]}")


@needs_analyzer
def test_a_stationary_cell_is_not_flagged(tmp_path):
    """The flag has to discriminate. A convergence test that fires on
    everything is the same as no test, and it would put a caveat on all
    50 cells."""
    root = str(tmp_path / "runs")
    _build_grid(root, drift=None)          # every cell stationary
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    r = _run_analyzer(root, out, "--allow-missing")
    blob = _blob(r, out)
    if "pofds3_" not in blob:
        pytest.skip(f"analyzer emitted nothing naming a cell:\n"
                    f"{_text(r)[-1500:]}")
    low = blob.lower()
    for m in ("not converged", "non-stationary", "still moving"):
        assert m not in low, (
            f"every cell in this grid is stationary, yet the analyzer "
            f"reports {m!r}")


# ------------------------------------------------------------- missing arms
@needs_analyzer
def test_a_missing_arm_is_a_hard_failure_not_a_hole_in_the_grid(tmp_path):
    """The failure this wave is most likely to actually hit: one of the
    46 jobs is held or evicted, the analyzer collects 45, and the figure
    renders a ladder with a rung missing. Nothing about the resulting
    plot says a cell is absent -- and the absent cell is disproportion-
    ately likely to be a high-lambda one, i.e. the end of the ladder the
    claim rests on."""
    root = str(tmp_path / "runs")
    missing = "pofds3_qwen7b_fwdlam8_eaopen_w0p5_k1_esopen_anch2_s0_r100"
    _build_grid(root, drop=missing)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    r = _run_analyzer(root, out)
    assert r.returncode != 0, (
        "analyzer exited 0 with an arm missing from the grid:\n"
        + _text(r)[-2000:])
    low = _text(r).lower()
    assert ("missing" in low or "hard fail" in low or "absent" in low
            or "incomplete" in low or "not found" in low), _text(r)[-2000:]


@needs_analyzer
def test_the_missing_arm_report_names_the_arm_that_is_missing(tmp_path):
    """Control for the test above. A gate that hard-fails on ANY tree it
    does not fully recognise would also "pass" that test while telling
    the reader nothing. The dropped tag must be named when it is absent,
    and must NOT be named when the grid is complete."""
    missing = "pofds3_qwen7b_fwdlam8_eaopen_w0p5_k1_esopen_anch2_s0_r100"
    short = str(tmp_path / "short")
    _build_grid(short, drop=missing)
    out_s = str(tmp_path / "out_short")
    os.makedirs(out_s, exist_ok=True)
    r_short = _run_analyzer(short, out_s)
    assert missing in _text(r_short), (
        "the analyzer refuses a short grid without naming the absent "
        f"arm:\n{_text(r_short)[-2000:]}")

    full = str(tmp_path / "full")
    _build_grid(full)
    out_f = str(tmp_path / "out_full")
    os.makedirs(out_f, exist_ok=True)
    r_full = _run_analyzer(full, out_f)
    absent = [l for l in _text(r_full).splitlines()
              if missing in l and ("absent" in l.lower()
                                   or "missing" in l.lower())]
    assert not absent, (
        f"a cell that IS present is reported absent: {absent[:3]}")
    # every one of the 46 production tags must be recognised, so that the
    # only thing standing between this tree and a clean run is the CPU
    # endpoint artifacts (which a synthetic tree does not fabricate).
    hard = [l for l in _text(r_full).splitlines()
            if "pofds3_" in l and "absent" in l.lower()]
    assert not hard, (
        "with a complete grid and a reuse manifest, no pofds3_ cell may "
        f"be reported absent:\n" + "\n".join(hard[:6]))


@needs_analyzer
def test_a_ten_round_scout_cannot_satisfy_a_production_arm(tmp_path):
    """pofdkd_* are 10-round scouts. If the analyzer's collection falls
    back to them when a 100-round arm is missing, the hole is filled
    with a run whose late window does not exist."""
    root = str(tmp_path / "runs")
    missing = "pofds3_qwen7b_fwdlam8_eaopen_w0p5_k1_esopen_anch2_s0_r100"
    _build_grid(root, drop=missing)
    _mk_run(root, "pofdkd_qwen7b_fwdlam8_eaopen_w0p5_l1_esopen_s0_r10",
            kl_beta=8.0, rounds=10)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    r = _run_analyzer(root, out)
    assert r.returncode != 0, (
        "a 10-round pofdkd_ scout satisfied a 100-round production arm:\n"
        + _text(r)[-2000:])


@needs_analyzer
def test_analyzer_does_not_silently_accept_a_partial_grid_by_default(mod):
    """If an --allow-missing escape hatch exists it must be OPT-IN. A
    default-on version of that flag is the same bug as no check."""
    h = _help()
    flag = next((f for f in ("--allow-missing", "--allow-partial")
                 if f in h), None)
    if flag is None:
        return
    src = inspect.getsource(mod)
    line = next((l for l in src.splitlines() if flag in l), "")
    ctx = src[max(0, src.find(line)):src.find(line) + 300]
    assert "store_true" in ctx, (
        f"{flag} must be a store_true OPT-IN; found: {ctx[:200]!r}")
    assert "default=True" not in ctx, (
        f"{flag} defaults to True -- a partial grid then renders silently")
