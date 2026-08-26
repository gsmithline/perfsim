#!/usr/bin/env python3
"""ANALYSIS for the Figure-4 anchor-tradeoff wave (fig4_anchor_tradeoff).

80 nominal cells = (model in {qwen7b, qwen3_8b}) x (es in {.05, .2}) x
(beta = W_PLAT in {0, .25, .5, .75, 1}) x (gamma = INNATE_LAMBDA in
{1, .5, .2, 0}); 60 are trained on the GPU, 20 are ALGEBRAIC DUPS that
resolve to a trained source cell through f4a_source():
  beta = 1  -> z = served exactly, gamma drops out: the (model, es, 1, 1)
               cell stands in for gamma in {.5, .2, 0};
  beta = 0  -> z = h exactly, the population never sees the model: the
               qwen3_8b cell stands in for qwen7b.
Every dup row in the CSV carries kind "dup" and the source tag it was
read from; its statistics are its source's statistics.

EVERY POPULATION STATISTIC IS POST-PEER: op_raw[t] is the END-OF-ROUND
state after the single Deffuant sweep of round t.  The late window is
the final ten rounds of the analysed horizon (rounds 21-30 of the base
30-round artifact; 51-60 / 91-100 of a _r60 / _r100 extension).  Each
trained cell is analysed at the LONGEST horizon present (_r100 > _r60 >
base) and the horizon used is recorded per row.

SETTLED (inherited from analyze_section4_gate.settle_verdict): a cell is
settled only if ALL hold on the round means of op_raw:
  (a) |final-5 half-split drift| (mean of the last 3 minus the mean of
      the first 2 of the final five) <= drift_tol (0.005)
  (b) |final-10 half-split drift| (final five minus the five before)
      <= drift_tol
  (c) range of the five late round means <= 2 * drift_tol (0.01)
An unsettled cell whose last 10 consecutive round-mean differences
alternate in sign on >= 70% of the steps is flagged CYCLIC.  Served
values are formatted .2f, so the 0.01 quantization floor applies to
every served-space comparison.

FROZEN (lambda = infinity) COMPARISON -- DISTRIBUTIONAL ONLY.  Every
trained cell has an offline replay of the model's constant zero-shot
served vector through the identical operator (replay_frozen_offline.py
-> notes/pofd/fig4_anchor/frozen/<f4a_frozen_name>, beta = 0 replays
named "shared").  The artifact must carry platform ==
"frozen_offline_replay" and w_plat / innate_k / eps_social / ab_sweeps
matching the cell; its pred_raw must be constant across rounds and
pred_raw[0] IS the entering-model (zero-shot prior) vector of the model.
The replay is NOT RNG-matched to the runner: it draws its peer pairs
from a CPU generator, the runner from a CUDA generator with the same
seed (same operator, same means; per-agent differences up to ~0.1-0.3,
mean |diff| ~0.01-0.04 even at beta = 0).  So NO equality is ever
asserted between a replay array and a run array; the comparison
columns are the final-mean difference, the two-sample KS statistic and
the L1 distance between the 50-bin histograms of the final populations,
plus the per-agent L1 (paired by agent index, but not RNG-matched --
read it as a distance, never as a mismatch).  The only replay-vs-run
array check is the INPUT population: innate must agree to 1e-4 (same
dataset / target / seed-invariant loader), which is data, not dynamics.

OUTPUTS (--out-dir, default notes/pofd/fig4_anchor)
  fig4_anchor_cells.csv               80 rows, one per nominal cell
  fig4_anchor_summary.json            tensor / window / tolerances /
                                      zsprior hashes / unsettled tags
  fig4_anchor_extension_request.json  the generator's manifest: a JSON
                                      list of {model, es, beta, gamma,
                                      rounds} for every UNSETTLED TRAINED
                                      cell (never a dup), rounds 60 (or
                                      100 if the cell is already at 60);
                                      copy to experiments/condor/ and the
                                      fig4_anchor_tradeoff_ext key runs
                                      exactly those cells

REFUSALS (numbers are never produced from a partial or ungated grid)
  1  the gate JSON is absent, not PASS, not this wave's full 80-cell
     verdict, or a trained tag is not PASS
  2  a trained cell is absent from every --run-root, or an artifact's
     horizon disagrees with its tag
  3  innate differs across artifacts (or between a replay and the wave
     beyond 1e-4), a frozen replay is absent or carries the wrong
     platform / dials, its served vector is not constant, or the
     zero-shot vector hash disagrees with the wave's canonical one
  4  --paper with any unsettled or cyclic cell

THE THREE NAMES, once: beta = W_PLAT (queue col wplat), gamma =
INNATE_LAMBDA (col lam), lambda = kl_beta (col beta) = 2 here.  The
homophily gamma is a different gamma and is always 0.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GEN = REPO / "experiments" / "condor" / "gen_pofd_sweep.py"
DEFAULT_RUN_ROOTS = (
    Path("/home/gsmithline/perfsim/runs/pokec_gated_lm"),
    REPO / "runs" / "pokec_gated_lm",
    REPO / "notes" / "pofd" / "cluster",
)
DEFAULT_OUT = REPO / "notes" / "pofd" / "fig4_anchor"
DEFAULT_FROZEN_DIR = DEFAULT_OUT / "frozen"

WINDOW = 10                  # late window: final ten rounds
N_LATE = 5                   # (a)/(c) look at the final five
DRIFT_TOL = 0.005            # opinion units
RANGE_TOL_MULT = 2           # (c): late-5 range <= 2 * tol
CYCLE_WINDOW = 10            # last 10 consecutive round-mean differences
CYCLE_ALTERNATION_MIN = 0.7  # ... alternating in sign on >= 70% of steps
SERVED_QUANT = 0.01          # labels are .2f
EXT_ROUNDS = (60, 100)       # allowed extension horizons, in order
FROZEN_PLATFORM = "frozen_offline_replay"
POP_UPDATE_V2 = "nested_ai_anchored_then_social_v2"
INNATE_TOL = 1e-6            # run vs run: one seed-invariant population
INNATE_TOL_REPLAY = 1e-4     # replay vs run: same INPUT data (not dynamics)
HIST_BINS = np.linspace(0.0, 1.0, 51)   # the plot's 50 shared bins
NA = "NA"
FROZEN_COMPARISON_NOTE = (
    "the lambda = infinity replay is NOT RNG-matched to the runner (CPU "
    "vs CUDA peer-pair generator, same seed, same operator, same means; "
    "per-agent differences up to ~0.1-0.3 even at beta = 0), so "
    "replay-vs-run comparisons are DISTRIBUTIONAL ONLY: final-mean "
    "difference, two-sample KS statistic and L1 between the 50-bin "
    "histograms of the final populations; the per-agent L1 is paired "
    "by agent index but not RNG-matched (a distance, never a mismatch). "
    "No equality between a replay array and a run array is asserted "
    "anywhere; the only replay-vs-run array check is the innate INPUT "
    "population (tolerance 1e-4), which is data, not dynamics.")

CELLS_CSV = "fig4_anchor_cells.csv"
SUMMARY_JSON = "fig4_anchor_summary.json"
EXT_JSON = "fig4_anchor_extension_request.json"

WITNESS_FIELDS = ("witness_lora_b_norm", "witness_probe_kl_fwd",
                  "witness_probe_argmax_agree")


# ------------------------------------------------------------------ grid
def load_gen(path=None):
    """The generator module, loaded by path (never as a package import).
    `path` lets a test point at a stand-in that implements the F4A API."""
    path = Path(path) if path else GEN
    spec = importlib.util.spec_from_file_location("_gen_f4a_analysis",
                                                  str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_f4a_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


class Grid:
    """The ONE accessor for the wave's grid API.  Everything the analyzer
    and the plot need from the generator's F4A block passes through
    here, so the scripts depend on the CONTRACT (f4a_cells / f4a_source /
    f4a_tag / f4a_frozen_name / F4A_ZSPRIOR / ...) and not on where it
    lives."""

    def __init__(self, gen):
        self.gen = gen
        self.key = str(gen.F4A_KEY)
        self.models = tuple(gen.F4A_MODELS)
        self.es = tuple(float(v) for v in gen.F4A_ES)
        self.betas = tuple(float(v) for v in gen.F4A_BETAS)
        self.gammas = tuple(float(v) for v in gen.F4A_GAMMAS)
        self.rounds = int(gen.F4A_ROUNDS)
        self.sweeps = int(gen.F4A_SWEEPS)
        self.seed = int(gen.F4A_SEED)
        self.beta0_model = str(gen.F4A_BETA0_MODEL)
        self.reused = dict(getattr(gen, "F4A_REUSED", {}) or {})
        self.zsprior = dict(gen.F4A_ZSPRIOR)
        self.zsprior_sha = dict(getattr(gen, "F4A_ZSPRIOR_SHA", {}) or {})
        fam = getattr(gen, "FAM_MODELS", None) or {}
        self.base_model = {m: fam[m]["base_model"] for m in self.models
                           if m in fam and "base_model" in fam[m]}

    def cells(self):
        """[(model, es, beta, gamma, kind, source)] in grid order."""
        out = []
        for (m, es, b, gm, kind, src) in self.gen.f4a_cells():
            out.append((str(m), float(es), float(b), float(gm), str(kind),
                        (str(src[0]), float(src[1]), float(src[2]),
                         float(src[3]))))
        return out

    def source(self, model, es, beta, gamma):
        s = self.gen.f4a_source(model, es, beta, gamma)
        return (str(s[0]), float(s[1]), float(s[2]), float(s[3]))

    def tag(self, model, es, beta, gamma, rounds=None):
        rounds = self.rounds if rounds is None else int(rounds)
        return str(self.gen.f4a_tag(model, es, beta, gamma, rounds=rounds))

    def base_tag(self, source):
        """The 30-round tag of a trained cell: the audited archived
        artifact when F4A_REUSED names one, else the wave's own tag."""
        return str(self.reused.get(tuple(source)) or self.tag(*source))

    def frozen_name(self, model, es, beta, gamma, rounds=None):
        rounds = self.rounds if rounds is None else int(rounds)
        m = "shared" if float(beta) == 0.0 else model
        return str(self.gen.f4a_frozen_name(m, es, beta, gamma,
                                            rounds=rounds))

    def num(self, v):
        return str(self.gen._num(v))


def _grid(gen=None, gen_path=None):
    """Grid from an already-loaded generator module, or from the
    generator at `gen_path` (default: the repo's)."""
    return Grid(gen if gen is not None else load_gen(gen_path))


# ---------------------------------------------------------------- gate
def gate_binds_wave(verdict, grid):
    """The gate JSON must be THIS wave's full production verdict: PASS,
    n_cells == 80, and every trained cell's tag present with status
    PASS.  Returns None when it binds, else the reason."""
    if not isinstance(verdict, dict) or not verdict.get("ok"):
        return "gate is not PASS"
    cells = grid.cells()
    n_expected = len(cells)
    if verdict.get("n_cells") != n_expected:
        return (f"gate verdict covers {verdict.get('n_cells')} cell(s), not "
                f"the {n_expected} production cells (smoke or stale?)")
    wave = verdict.get("wave")
    if wave is not None and grid.key not in str(wave):
        return f"gate verdict is for wave {wave!r}, not {grid.key!r}"
    want = {grid.base_tag(src) for (_, _, _, _, kind, src) in cells
            if kind == "gpu"}
    got = {}
    for c in verdict.get("cells") or []:
        if isinstance(c, dict) and c.get("tag"):
            got[str(c["tag"])] = c.get("status")
    missing = sorted(t for t in want if got.get(t) != "PASS")
    if missing:
        return (f"{len(missing)} trained tag(s) not PASS in the gate "
                f"(first: {missing[0]})")
    return None


# ------------------------------------------------------------ helpers
def _find(tag, roots):
    for root in roots:
        p = Path(root) / tag / "trajectory.pt"
        if p.exists():
            return p
    return None


def _np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _sha_f32(vec):
    return hashlib.sha256(
        np.ascontiguousarray(_np(vec).astype(np.float32)).tobytes()
    ).hexdigest()


def _load(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def half_split(idx):
    """The inherited 2-vs-3 split of the final five: first 2 / last 3."""
    idx = list(idx)
    h = len(idx) // 2
    return idx[:h], idx[h:]


def late_stats(op, horizon, window=WINDOW):
    """Late-window statistics on the POST-PEER round means of op_raw
    over rounds [horizon-window+1, horizon] (1-based)."""
    op = _np(op)[:horizon]
    means = op.mean(axis=1)
    tail = means[-window:]
    late_idx = list(range(horizon - N_LATE, horizon))
    h1, h2 = half_split(late_idx)
    drift5 = float(means[h2].mean() - means[h1].mean())
    prev_idx = [t - N_LATE for t in late_idx]
    drift10 = float(means[late_idx].mean() - means[prev_idx].mean())
    late5 = means[late_idx]
    late5_range = float(late5.max() - late5.min())
    last = horizon - 1
    diffs = np.array([means[t] - means[t - 1]
                      for t in range(last - CYCLE_WINDOW + 1, last + 1)])
    steps = len(diffs) - 1
    flips = int(sum(1 for i in range(steps) if diffs[i] * diffs[i + 1] < 0))
    return {
        "equilibrium_mean": float(tail.mean()),
        "final_mean": float(means[horizon - 1]),
        "final_sd": float(op[horizon - 1].std()),
        "drift5": drift5,
        "drift10": drift10,
        "late5_range": late5_range,
        "alternating_frac": float(flips / steps),
        "window_rounds": [horizon - window + 1, horizon],
    }


def settled(stats, tol=DRIFT_TOL):
    return bool(abs(stats["drift5"]) <= tol
                and abs(stats["drift10"]) <= tol
                and stats["late5_range"] <= RANGE_TOL_MULT * tol)


def cyclic(stats, tol=DRIFT_TOL):
    return bool((not settled(stats, tol))
                and stats["alternating_frac"] >= CYCLE_ALTERNATION_MIN)


def outcome_of(stats, horizon, tol=DRIFT_TOL):
    if settled(stats, tol):
        return "equilibrium"
    nxt = [r for r in EXT_ROUNDS if r > horizon]
    if nxt:
        return f"extend_to_{nxt[0]}"
    return "cyclic_long_run" if cyclic(stats, tol) else "drifting_long_run"


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic sup |F_a - F_b| (no
    scipy on the login node)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(fa - fb).max())


def hist_l1(a, b, bins=HIST_BINS):
    """L1 distance between the normalised fixed-width histograms of two
    populations on the plot's 50 shared bins (0 = identical bin masses,
    2 = disjoint)."""
    ca, _ = np.histogram(np.clip(a, 0.0, 1.0), bins=bins)
    cb, _ = np.histogram(np.clip(b, 0.0, 1.0), bins=bins)
    return float(np.abs(ca / ca.sum() - cb / cb.sum()).sum())


def frozen_comparison(final, frozen_final):
    """Distributional distances between a run's and a replay's final
    post-peer populations (see FROZEN_COMPARISON_NOTE)."""
    final = np.asarray(final, dtype=np.float64)
    frozen_final = np.asarray(frozen_final, dtype=np.float64)
    diff = final - frozen_final
    return {
        "frozen_final_mean": float(frozen_final.mean()),
        "frozen_final_sd": float(frozen_final.std()),
        "abs_final_mean_diff_vs_frozen": abs(float(final.mean())
                                             - float(frozen_final.mean())),
        "ks_final_vs_frozen": ks_statistic(final, frozen_final),
        "hist_l1_final_vs_frozen": hist_l1(final, frozen_final),
        "l1_final_vs_frozen": float(np.abs(diff).sum()),
        "mean_abs_final_vs_frozen": float(np.abs(diff).mean()),
    }


def served_cardinality(pred, horizon):
    """Greedy cardinality: distinct served values in the final round, the
    minimum over rounds, and the final-round modal share."""
    pred = _np(pred)[:horizon]
    per_round = [len(np.unique(np.round(pred[t], 8)))
                 for t in range(pred.shape[0])]
    values, counts = np.unique(np.round(pred[-1], 8), return_counts=True)
    return {"served_distinct_final": int(per_round[-1]),
            "served_distinct_min": int(min(per_round)),
            "served_modal_share_final": float(counts.max() / counts.sum())}


def witness_summary(run_dir, horizon):
    """min witness_lora_b_norm, min witness_probe_kl_fwd, mean
    witness_probe_argmax_agree over the telemetry rounds < horizon;
    NA when the telemetry or a field is absent (the gate already
    verified the witness; this is a summary, not a second gate)."""
    out = {"witness_lora_b_norm_min": NA, "witness_probe_kl_fwd_min": NA,
           "witness_probe_argmax_agree_mean": NA, "witness_rounds": 0}
    p = Path(run_dir) / "telemetry.json"
    if not p.exists():
        return out
    vals = {k: [] for k in WITNESS_FIELDS}
    n = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "round" in rec and int(rec["round"]) >= horizon:
            continue
        if not any(k in rec for k in WITNESS_FIELDS):
            continue
        n += 1
        for k in WITNESS_FIELDS:
            if rec.get(k) is not None:
                vals[k].append(float(rec[k]))
    out["witness_rounds"] = n
    if vals["witness_lora_b_norm"]:
        out["witness_lora_b_norm_min"] = min(vals["witness_lora_b_norm"])
    if vals["witness_probe_kl_fwd"]:
        out["witness_probe_kl_fwd_min"] = min(vals["witness_probe_kl_fwd"])
    if vals["witness_probe_argmax_agree"]:
        out["witness_probe_argmax_agree_mean"] = float(
            np.mean(vals["witness_probe_argmax_agree"]))
    return out


def find_trained(source, grid, roots):
    """The LONGEST-horizon artifact of a trained cell: (_r100, 100) >
    (_r60, 60) > (base tag, 30).  -> (path, tag, horizon) or (None,
    base_tag, 0)."""
    base = grid.base_tag(source)
    candidates = [(grid.tag(*source, rounds=r), r)
                  for r in sorted(EXT_ROUNDS, reverse=True)]
    candidates.append((base, grid.rounds))
    for tag, horizon in candidates:
        p = _find(tag, roots)
        if p is not None:
            return p, tag, horizon
    return None, base, 0


def find_frozen(source, grid, frozen_dir, horizon):
    """The frozen replay of a trained cell at the cell's horizon when it
    exists, else the base 30-round replay.  -> (path, name, rounds) or
    (None, base_name, 0)."""
    model, es, beta, gamma = source
    tried = []
    for r in ([horizon] if horizon != grid.rounds else []) + [grid.rounds]:
        name = grid.frozen_name(model, es, beta, gamma, rounds=r)
        tried.append(name)
        p = Path(frozen_dir) / name
        if p.exists():
            return p, name, r
    return None, tried[-1], 0


def check_frozen_config(cfg, source, grid, name):
    """Problems with a frozen replay's config, as strings."""
    model, es, beta, gamma = source
    bad = []
    cfg = cfg or {}
    if cfg.get("platform") != FROZEN_PLATFORM:
        bad.append(f"{name}: platform={cfg.get('platform')!r} != "
                   f"{FROZEN_PLATFORM!r}")
    pu = cfg.get("population_update")
    if pu != POP_UPDATE_V2:
        bad.append(f"{name}: population_update={pu!r} != {POP_UPDATE_V2!r}")
    for field, want in (("w_plat", beta), ("innate_k", gamma),
                        ("eps_social", es), ("ab_sweeps", grid.sweeps)):
        got = cfg.get(field)
        if got is None or abs(float(got) - float(want)) > 1e-9:
            bad.append(f"{name}: config {field}={got!r} != {want!r}")
    if beta > 0.0 and model in grid.base_model:
        bm = cfg.get("base_model")
        if bm != grid.base_model[model]:
            bad.append(f"{name}: base_model={bm!r} != "
                       f"{grid.base_model[model]!r} ({model})")
    return bad


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="per-cell post-peer outcomes, settling verdicts, "
                    "frozen-replay comparison and extension manifest for "
                    "the Figure-4 anchor-tradeoff wave")
    ap.add_argument("--run-root", action="append", default=None)
    ap.add_argument("--gate-json", required=True,
                    help="check_fig4_anchor.py --json verdict (must be the "
                         "full 80-cell PASS)")
    ap.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--paper", action="store_true",
                    help="exit 4 if any cell is unsettled or cyclic")
    ap.add_argument("--drift-tol", type=float, default=DRIFT_TOL)
    ap.add_argument("--sweeps", type=int, default=100, choices=(100, 1),
                    help="which sibling wave: 100 (primary) or 1 (dirs default to *_sw1)")
    ap.add_argument("--gen", default=None,
                    help="path of the generator module (default: the "
                         "repo's gen_pofd_sweep.py)")
    args = ap.parse_args(argv)
    if args.sweeps != 100:
        if args.out_dir == str(DEFAULT_OUT):
            args.out_dir = str(DEFAULT_OUT.parent / f"fig4_anchor_sw{args.sweeps}")
        if args.frozen_dir == str(DEFAULT_FROZEN_DIR):
            args.frozen_dir = str(DEFAULT_OUT / f"frozen_sw{args.sweeps}")
    tol = float(args.drift_tol)

    gate_path = Path(args.gate_json)
    if not gate_path.exists():
        print(f"[analyze_f4a] REFUSING: gate JSON {gate_path} absent",
              file=sys.stderr)
        return 1
    try:
        verdict = json.loads(gate_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[analyze_f4a] REFUSING: gate JSON unreadable: {e}",
              file=sys.stderr)
        return 1
    _gen_mod = load_gen(args.gen)
    if hasattr(_gen_mod, "f4a_set_variant"):
        _gen_mod.f4a_set_variant(args.sweeps)
    grid = _grid(gen=_gen_mod)
    why = gate_binds_wave(verdict, grid)
    if why:
        print(f"[analyze_f4a] REFUSING: {why}", file=sys.stderr)
        return 1

    roots = [Path(r) for r in (args.run_root or DEFAULT_RUN_ROOTS)]
    frozen_dir = Path(args.frozen_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = grid.cells()
    sources = []
    for c in cells:
        if c[5] not in sources:
            sources.append(c[5])

    # ---- trained cells: longest horizon, stats -------------------------
    trained, missing, horizon_bad = {}, [], []
    for src in sources:
        path, tag, horizon = find_trained(src, grid, roots)
        if path is None:
            missing.append(tag)
            continue
        d = _load(path)
        op = _np(d["op_raw"])
        pred = _np(d["pred_raw"])
        tw = _np(d["twin_raw"])
        cfg_rounds = (d.get("config") or {}).get("n_rounds")
        if (op.shape[0] != horizon or pred.shape[0] != horizon
                or tw.shape[0] != horizon
                or (cfg_rounds is not None and int(cfg_rounds) != horizon)):
            horizon_bad.append(f"{tag}: op/pred/twin hold {op.shape[0]}/"
                               f"{pred.shape[0]}/{tw.shape[0]} rounds, "
                               f"config n_rounds={cfg_rounds!r}, the tag "
                               f"says {horizon}")
            continue
        st = late_stats(op, horizon)
        st.update(served_cardinality(pred, horizon))
        st.update(witness_summary(path.parent, horizon))
        st.update({
            "path": str(path), "tag": tag, "horizon": horizon,
            "innate": _np(d["innate"]),
            "final": op[horizon - 1].copy(),
            "twin_final_mean": float(tw[horizon - 1].mean()),
            "git_sha": (d.get("config") or {}).get("git_sha"),
        })
        trained[src] = st
        del d
    if missing or horizon_bad:
        print(f"[analyze_f4a] REFUSING: {len(missing)} trained cell(s) "
              f"absent and {len(horizon_bad)} at the wrong horizon",
              file=sys.stderr)
        for t in missing:
            print(f"    ABSENT {t}", file=sys.stderr)
        for t in horizon_bad:
            print(f"    HORIZON {t}", file=sys.stderr)
        return 2

    # ---- innate: one population across the wave -----------------------
    ref_src = sources[0]
    ref_innate = trained[ref_src]["innate"]
    innate_sha = _sha_f32(ref_innate)
    for src, st in trained.items():
        if (st["innate"].shape != ref_innate.shape
                or float(np.abs(st["innate"] - ref_innate).max()) > INNATE_TOL):
            print(f"[analyze_f4a] REFUSING: innate of {st['tag']} differs "
                  f"from {trained[ref_src]['tag']} (max |diff| > "
                  f"{INNATE_TOL:g}) -- not one population", file=sys.stderr)
            return 3
    innate_mean = float(ref_innate.mean())

    # ---- frozen replays: config, constant served vector, final pop ----
    frozen, frozen_bad = {}, []
    for src in sources:
        st = trained[src]
        path, name, fr_rounds = find_frozen(src, grid, frozen_dir,
                                            st["horizon"])
        if path is None:
            frozen_bad.append(f"ABSENT {frozen_dir / name}")
            continue
        d = _load(path)
        cfg = d.get("config") or {}
        frozen_bad += check_frozen_config(cfg, src, grid, name)
        fop = _np(d["op_raw"])
        fpred = _np(d["pred_raw"])
        finn = _np(d["innate"])
        if fop.shape[0] < fr_rounds or fpred.shape[0] < 1:
            frozen_bad.append(f"{name}: holds {fop.shape[0]} rounds, "
                              f"expected {fr_rounds}")
            continue
        if not np.array_equal(fpred, np.broadcast_to(fpred[0], fpred.shape)):
            nvary = int((fpred != fpred[0]).any(axis=0).sum())
            frozen_bad.append(f"{name}: served vector NOT constant across "
                              f"rounds ({nvary} agents vary)")
            continue
        # the INPUT population only (same dataset / target); never an
        # op_raw / twin_raw / pred_raw equality -- the replay is not
        # RNG-matched to the runner (FROZEN_COMPARISON_NOTE)
        if (finn.shape != ref_innate.shape
                or float(np.abs(finn - ref_innate).max()) > INNATE_TOL_REPLAY):
            frozen_bad.append(f"{name}: innate (input population) differs "
                              f"from the wave's beyond {INNATE_TOL_REPLAY:g}")
            continue
        frozen[src] = {
            "name": name, "path": str(path), "rounds": fr_rounds,
            "final": fop[fr_rounds - 1].copy(),
            "pred": fpred[0].astype(np.float32),
            "pred_sha": _sha_f32(fpred[0]),
            "base_model": cfg.get("base_model"),
        }
        del d
    if frozen_bad:
        print(f"[analyze_f4a] REFUSING: {len(frozen_bad)} frozen-replay "
              f"problem(s)", file=sys.stderr)
        for b in frozen_bad:
            print(f"    {b}", file=sys.stderr)
        return 3

    # ---- entering-model vectors: one constant vector per model --------
    zs, zs_bad = {}, []
    for model in grid.models:
        shas = {}
        for src, fr in frozen.items():
            if src[0] == model and src[2] > 0.0:
                shas.setdefault(fr["pred_sha"], []).append(fr["name"])
                zs.setdefault(model, fr["pred"])
        if not shas:
            zs_bad.append(f"{model}: no beta > 0 frozen replay carries its "
                          f"zero-shot vector")
            continue
        if len(shas) > 1:
            zs_bad.append(f"{model}: {len(shas)} different served vectors "
                          f"across its frozen replays: "
                          + "; ".join(f"{s[:12]} x{len(v)}"
                                      for s, v in shas.items()))
            continue
        sha = next(iter(shas))
        want = grid.zsprior_sha.get(model)
        if want and sha != want:
            zs_bad.append(f"{model}: zero-shot vector sha256 {sha} != "
                          f"canonical {want}")
    for src, fr in frozen.items():
        if src[2] == 0.0:
            owners = [m for m, v in zs.items() if _sha_f32(v) == fr["pred_sha"]]
            if not owners:
                zs_bad.append(f"{fr['name']}: shared beta=0 replay carries "
                              f"a served vector that is neither model's "
                              f"zero-shot vector")
            fr["pred_model"] = owners[0] if owners else None
    if zs_bad:
        print("[analyze_f4a] REFUSING: zero-shot vector inconsistency",
              file=sys.stderr)
        for b in zs_bad:
            print(f"    {b}", file=sys.stderr)
        return 3
    zs_mean = {m: float(v.mean()) for m, v in zs.items()}

    # ---- rows ----------------------------------------------------------
    rows, extensions, not_ext = [], [], []
    for (model, es, beta, gamma, kind, src) in cells:
        st = trained[src]
        fr = frozen[src]
        horizon = st["horizon"]
        is_settled = settled(st, tol)
        is_cyclic = cyclic(st, tol)
        oc = outcome_of(st, horizon, tol)
        fc = frozen_comparison(st["final"], fr["final"])
        row = {
            "model": model,
            "es": f"{es:g}",
            "beta": f"{beta:g}",
            "gamma": f"{gamma:g}",
            "kind": kind,
            "source_tag": st["tag"],
            "path": st["path"],
            "horizon": horizon,
            "window_rounds": f"{st['window_rounds'][0]}-"
                             f"{st['window_rounds'][1]}",
            "final_mean": f"{st['final_mean']:.8f}",
            "final_sd": f"{st['final_sd']:.8f}",
            "innate_mean": f"{innate_mean:.8f}",
            "entering_mean": f"{zs_mean[model]:.8f}",
            "twin_final_mean": f"{st['twin_final_mean']:.8f}",
            "final_minus_innate": f"{st['final_mean'] - innate_mean:.8f}",
            "final_minus_entering":
                f"{st['final_mean'] - zs_mean[model]:.8f}",
            "equilibrium_mean": f"{st['equilibrium_mean']:.8f}",
            "drift5": f"{st['drift5']:.8f}",
            "drift10": f"{st['drift10']:.8f}",
            "late5_range": f"{st['late5_range']:.8f}",
            "alternating_frac": f"{st['alternating_frac']:.4f}",
            "settled": is_settled,
            "cyclic": is_cyclic,
            "outcome": oc,
            "served_distinct_final": st["served_distinct_final"],
            "served_distinct_min": st["served_distinct_min"],
            "served_modal_share_final":
                f"{st['served_modal_share_final']:.8f}",
            "witness_lora_b_norm_min": st["witness_lora_b_norm_min"],
            "witness_probe_kl_fwd_min": st["witness_probe_kl_fwd_min"],
            "witness_probe_argmax_agree_mean":
                st["witness_probe_argmax_agree_mean"],
            "witness_rounds": st["witness_rounds"],
            "frozen_name": fr["name"],
            "frozen_rounds": fr["rounds"],
            "frozen_final_mean": f"{fc['frozen_final_mean']:.8f}",
            "frozen_final_sd": f"{fc['frozen_final_sd']:.8f}",
            "abs_final_mean_diff_vs_frozen":
                f"{fc['abs_final_mean_diff_vs_frozen']:.8f}",
            "ks_final_vs_frozen": f"{fc['ks_final_vs_frozen']:.8f}",
            "hist_l1_final_vs_frozen": f"{fc['hist_l1_final_vs_frozen']:.8f}",
            "l1_final_vs_frozen": f"{fc['l1_final_vs_frozen']:.8f}",
            "mean_abs_final_vs_frozen":
                f"{fc['mean_abs_final_vs_frozen']:.8f}",
            "git_sha": st["git_sha"],
        }
        rows.append(row)
        if kind == "gpu" and not is_settled:
            if oc.startswith("extend_to_"):
                extensions.append({"model": model, "es": es, "beta": beta,
                                   "gamma": gamma,
                                   "rounds": int(oc.rsplit("_", 1)[1])})
            else:
                not_ext.append({"model": model, "es": es, "beta": beta,
                                "gamma": gamma, "horizon": horizon,
                                "outcome": oc})

    cells_csv = out / CELLS_CSV
    with cells_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ext_path = out / EXT_JSON
    ext_path.write_text(json.dumps(extensions, indent=2) + "\n")

    unsettled = [r for r in rows if not r["settled"]]
    cyclic_rows = [r for r in rows if r["cyclic"]]
    gate_shas = sorted({str(s) for s in (verdict.get("git_sha") or [])
                        if s}) or sorted({str(c.get("git_sha"))
                                          for c in verdict.get("cells", [])
                                          if isinstance(c, dict)
                                          and c.get("git_sha")})
    summary = {
        "gated": True,
        "wave": grid.key,
        "gate_json": str(gate_path),
        "gate_n_cells": verdict.get("n_cells"),
        "git_sha": gate_shas,
        "n_cells": len(rows),
        "n_trained": sum(1 for r in rows if r["kind"] == "gpu"),
        "n_dup": sum(1 for r in rows if r["kind"] == "dup"),
        "models": list(grid.models),
        "es": list(grid.es),
        "betas": list(grid.betas),
        "gammas": list(grid.gammas),
        "seed": grid.seed,
        "rounds": grid.rounds,
        "ab_sweeps": grid.sweeps,
        "postpeer": True,
        "tensor": "op_raw (end-of-round, post-peer)",
        "window": WINDOW,
        "window_rounds": [grid.rounds - WINDOW + 1, grid.rounds],
        "window_note": ("final ten rounds of the analysed horizon; a "
                        "_r60 / _r100 extension is read on ITS final ten"),
        "drift_tol": tol,
        "late_range_tol": RANGE_TOL_MULT * tol,
        "cycle_window": CYCLE_WINDOW,
        "cycle_alternation_min": CYCLE_ALTERNATION_MIN,
        "settled_rule": ("|final-5 half-split drift| <= drift_tol AND "
                         "|final-10 half-split drift| <= drift_tol AND "
                         "late-5 range <= late_range_tol"),
        "served_quantization": SERVED_QUANT,
        "innate_mean": innate_mean,
        "innate_sha256": innate_sha,
        "frozen_dir": str(frozen_dir),
        "frozen_platform": FROZEN_PLATFORM,
        "frozen_comparison": "distributional only",
        "frozen_comparison_columns": [
            "frozen_final_mean", "frozen_final_sd",
            "abs_final_mean_diff_vs_frozen", "ks_final_vs_frozen",
            "hist_l1_final_vs_frozen", "l1_final_vs_frozen",
            "mean_abs_final_vs_frozen"],
        "frozen_comparison_note": FROZEN_COMPARISON_NOTE,
        "frozen_hist_bins": int(len(HIST_BINS) - 1),
        "entering_model_vector": ("pred_raw[0] of the model's frozen "
                                  "replay, asserted constant across the "
                                  "replay's rounds and identical across "
                                  "the model's replays"),
        "zsprior": {
            m: {"source_tag": grid.zsprior.get(m),
                "sha256": _sha_f32(zs[m]),
                "expected_sha256": grid.zsprior_sha.get(m),
                "mean": zs_mean[m],
                "sd": float(zs[m].std()),
                "n_distinct": int(len(np.unique(np.round(zs[m], 8)))),
                "n_frozen_artifacts": sum(1 for s in frozen
                                          if s[0] == m and s[2] > 0.0)}
            for m in grid.models},
        "horizons": sorted({int(r["horizon"]) for r in rows}),
        "n_unsettled": len(unsettled),
        "n_cyclic": len(cyclic_rows),
        "unsettled_tags": sorted({r["source_tag"] for r in unsettled}),
        "cyclic_tags": sorted({r["source_tag"] for r in cyclic_rows}),
        "extension_request": {"path": str(ext_path),
                              "n_cells": len(extensions),
                              "format": "[{model, es, beta, gamma, rounds}]"
                                        " -- trained cells only, never dups",
                              "cells": extensions},
        "not_extendable": not_ext,
        "cells_csv": str(cells_csv),
    }
    (out / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")

    # ---- console -------------------------------------------------------
    print("=" * 112)
    print(f"FIGURE-4 ANCHOR TRADE-OFF -- {len(rows)} cells "
          f"({summary['n_trained']} trained + {summary['n_dup']} dup), "
          f"POST-PEER, window last {WINDOW} rounds, drift tol {tol:g}, "
          f"late-5 range tol {RANGE_TOL_MULT * tol:g}")
    print("=" * 112)
    print(f"innate mean {innate_mean:.6f}; entering-model means: "
          + ", ".join(f"{m} {zs_mean[m]:.6f} ({summary['zsprior'][m]['n_distinct']} "
                      f"distinct)" for m in grid.models))
    print(f"{'model':<9}{'es':>5}{'beta':>6}{'gamma':>6}{'kind':>5}"
          f"{'hor':>4}{'final':>9}{'sd':>8}{'frozen':>9}{'KS':>7}"
          f"{'twin':>9}{'drift10':>9}{'rng5':>8}  outcome")
    print("-" * 112)
    for r in rows:
        print(f"{r['model']:<9}{r['es']:>5}{r['beta']:>6}{r['gamma']:>6}"
              f"{r['kind']:>5}{r['horizon']:>4}"
              f"{float(r['final_mean']):>9.4f}{float(r['final_sd']):>8.4f}"
              f"{float(r['frozen_final_mean']):>9.4f}"
              f"{float(r['ks_final_vs_frozen']):>7.3f}"
              f"{float(r['twin_final_mean']):>9.4f}"
              f"{float(r['drift10']):>+9.4f}{float(r['late5_range']):>8.4f}"
              f"  {r['outcome']}{' CYCLIC' if r['cyclic'] else ''}")
    print("(frozen = lambda-infinity replay, NOT RNG-matched: compare "
          "distributions, never agents)")
    print("=" * 112)
    print(f"[analyze_f4a] {len(rows) - len(unsettled)} settled, "
          f"{len(unsettled)} unsettled ({len(cyclic_rows)} cyclic); "
          f"{len(extensions)} extension request(s)")
    print(f"[analyze_f4a] wrote {cells_csv}")
    print(f"[analyze_f4a] wrote {out / SUMMARY_JSON}")
    print(f"[analyze_f4a] wrote {ext_path}"
          + (" -- copy to experiments/condor/fig4_anchor_extension_request"
             ".json, commit, and submit fig4_anchor_tradeoff_ext"
             if extensions else " (empty: nothing to extend)"))
    if not_ext:
        print(f"[analyze_f4a] NOTE {len(not_ext)} trained cell(s) unsettled "
              f"at the last allowed horizon: long-run outcomes, never "
              f"equilibria")
    if args.paper and (unsettled or cyclic_rows):
        print(f"[analyze_f4a] PAPER GATE FAIL: {len(unsettled)} unsettled "
              f"and {len(cyclic_rows)} cyclic cell(s)", file=sys.stderr)
        for r in unsettled:
            print(f"    {r['kind']} {r['model']} es={r['es']} "
                  f"beta={r['beta']} gamma={r['gamma']} -> {r['outcome']}"
                  f"{' CYCLIC' if r['cyclic'] else ''} ({r['source_tag']})",
                  file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
