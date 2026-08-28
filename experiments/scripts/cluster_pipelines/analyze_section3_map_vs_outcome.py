#!/usr/bin/env python3
"""Does WHERE a model starts explain WHERE its population ends up?

Reviewer question for Figure 3(a): the six models reach visibly
different equilibria -- but is that just inherited from how differently
they predict in the first place?  This compares, over all 15 model
pairs, how far apart two models' INITIAL prediction maps are with how
far apart their FINAL post-peer populations are.

THE X AXIS: the initial zero-shot prediction map.  Taken from the
FROZEN personal-history ICL wave's round 0 (pofds3i_<model>_d8_greedy_
..._r30, pred_raw[0]).  Those runs never take a gradient step, so
round 0 is the base model's own map over the 723 MovieLens/Action
agents before any adaptation and before any feedback.  It is verified
here to be BIT-IDENTICAL across seeds {0,42,43} for every model -- as
it must be, since the weights are frozen, decoding is greedy and the
innate vector is seed-invariant.  The initial map therefore carries no
seed uncertainty; all of it sits on the y axis.

  primary   agent-paired mean |p_A(i) - p_B(i)| over matched agents
  secondary equal-mass W1 between the two maps (distribution only),
            reported alongside because it is the same functional form
            as the y axis

A SECOND X AXIS IS ALSO REPORTED: the map Figure 3(a) actually deploys
at t=0, i.e. pred_raw[0] of the SFT runs themselves, which is taken
AFTER round 0's supervised step on pristine labels.  That map is
seed-dependent, so it is averaged over the paired seeds.  It is the
right x if "initial" is read as "first deployed" rather than
"zero-shot"; both are in the CSV and the JSON.

THE Y AXIS: paired-seed equal-mass W1 between round-30 post-peer
populations, strict corrected runs only.  Recomputed here from the
trajectories with the same definition analyze_section3_model_pairwise_w1
.py uses, so that Mistral's seed set can be varied; agreement with that
script's published CSV is ASSERTED on the seeds it used.

MISTRAL -- ITS ACTUAL VALID SEED COUNT IS ONE, NOT TWO.  Re-gating the
Figure-3(a) SFT wave with check_section3_model_equilibria.py (2026-08-28)
fails Mistral at seeds 0 AND 43; only seed 42 passes:

  s0   round 1: all 723 generations malformed (parse_fail_frac 1.0), so
       every agent was served the parser's 0.5 default
  s43  round 25: all 723 generations are '58 (58' -- again 100% served
       the 0.5 default.  From round 26 the model, now trained on that
       round's ~0.5 population, genuinely emits '0.50 (' with
       parse_fail_frac 0, so the later rounds pass a per-round parse
       gate while carrying the injected default forward.  The final
       population mean is 0.500000.
  s42  clean in every round; final population mean 0.540000.

Seed 43's round-30 population is therefore a downstream consequence of
a parser failure, not a model outcome, and it sits inside the late
window.  The DEFAULT here is --mistral-seeds 42: Mistral's five pairs
rest on ONE seed, are reported as a single observation with NO
interval, and are drawn distinctly.  --mistral-seeds 42,43 reproduces
the published model_pairwise_w1_round30.csv exactly (asserted), and
both Spearman values are reported so the reader can see whether the
exclusion changes the conclusion.  (Mistral's round-0 deployed map is
additionally degenerate -- a constant 1.0 for all 723 agents.)

THE 15 PAIRS ARE NOT INDEPENDENT.  Each of the 6 models appears in 5
pairs, so the pairs share underlying draws.  Spearman's rho and the
fitted line are reported as DESCRIPTIVE SUMMARIES ONLY: no p-value and
no confidence band is computed, because the independence those
procedures assume does not hold here.  A leave-one-model-out sweep (6
refits, each dropping a model and its 5 pairs) is reported instead, as
an honest sensitivity to that dependence.

Figures carry NO title (house rule); the caption block is written
beside the PDF.

  python analyze_section3_map_vs_outcome.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-mvo"))

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

MODELS = ("qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
          "ministral8b")
LABEL = {"qwen7b": "Qwen 2.5", "qwen3_8b": "Qwen 3", "olmo7b": "OLMo 2",
         "olmo3_7b": "OLMo 3", "mistral7b": "Mistral",
         "ministral8b": "Ministral"}
SHORT = {"qwen7b": "Q2.5", "qwen3_8b": "Q3", "olmo7b": "O2",
         "olmo3_7b": "O3", "mistral7b": "Mi", "ministral8b": "Mn"}
FAMILY = {"qwen7b": "Qwen", "qwen3_8b": "Qwen", "olmo7b": "OLMo",
          "olmo3_7b": "OLMo", "mistral7b": "Mistral",
          "ministral8b": "Mistral"}
ICL_TAG = ("pofds3i_{model}_d8_greedy_sw100_eaopen_w1_k1_esopen_anch2"
           "_s{seed}_r30")
SEEDS = (0, 42, 43)
N_AGENTS = 723

WITHIN_C = "#4c72b0"
ACROSS_C = "#c44e52"
INK = "#202328"


def load_zero_shot(root: Path) -> dict[str, np.ndarray]:
    """Round-0 prediction map of each FROZEN ICL run, verified
    bit-identical across seeds."""
    maps, errs = {}, []
    for m in MODELS:
        vecs = []
        for s in SEEDS:
            p = root / ICL_TAG.format(model=m, seed=s) / "trajectory.pt"
            if not p.exists():
                errs.append(f"{m} s{s}: {p} absent")
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            cfg = d.get("config", {}) or {}
            if cfg.get("training_style") != "frozen":
                errs.append(f"{m} s{s}: training_style="
                            f"{cfg.get('training_style')!r} (want frozen -- "
                            f"a trained round 0 is not a zero-shot map)")
            if cfg.get("do_sample") not in (False, 0, None):
                errs.append(f"{m} s{s}: do_sample={cfg.get('do_sample')!r}")
            v = d["pred_raw"][0].float().numpy()
            if v.shape != (N_AGENTS,):
                errs.append(f"{m} s{s}: map shape {v.shape}")
            vecs.append(v)
        if not vecs:
            continue
        if not all(np.array_equal(vecs[0], v) for v in vecs[1:]):
            errs.append(f"{m}: round-0 frozen map is NOT identical across "
                        f"seeds -- the x axis would carry seed noise")
        maps[m] = vecs[0]
    return maps, errs


def load_deployed(cells_csv: Path, root: Path):
    """Round-0 map of the Figure-3(a) SFT runs (post round-0 SFT), by
    seed. Mistral comes from the strict-parse cells."""
    by = {m: {} for m in MODELS}
    with cells_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            m = r["model"]
            if m not in by or m == "mistral7b":
                continue
            d = torch.load(REPO / r["path"], map_location="cpu",
                           weights_only=False)
            by[m][int(r["seed"])] = d["pred_raw"][0].float().numpy()
    for s in (42, 43):
        p = (root / f"pofds3m_mistral7b_fwdlam2_sw100_eaopen_w1_k1_esopen"
                    f"_anch2_pstrict_s{s}_r30" / "trajectory.pt")
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            by["mistral7b"][s] = d["pred_raw"][0].float().numpy()
    return by


MISTRAL_STRICT = ("pofds3m_mistral7b_fwdlam2_sw100_eaopen_w1_k1_esopen"
                  "_anch2_pstrict_s{seed}_r30")
T95_DF2 = 4.302652729911275


def load_final_populations(cells_csv: Path, root: Path, mistral_seeds):
    """Round-30 post-peer population per (model, seed). Non-Mistral cells
    come from the Section-3 analysis CSV; Mistral comes from its
    strict-parse tags, on whichever seeds the caller declares valid."""
    by = {m: {} for m in MODELS}
    with cells_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            m = r["model"]
            if m not in by or m == "mistral7b":
                continue
            d = torch.load(REPO / r["path"], map_location="cpu",
                           weights_only=False)
            by[m][int(r["seed"])] = np.asarray(
                d["op_raw"][-1].float().numpy(), dtype=float).reshape(-1)
    for s in mistral_seeds:
        p = root / MISTRAL_STRICT.format(seed=s) / "trajectory.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        by["mistral7b"][s] = np.asarray(
            d["op_raw"][-1].float().numpy(), dtype=float).reshape(-1)
    return by


def outcome_row(a, b, pops):
    seeds = sorted(set(pops[a]) & set(pops[b]))
    vals = np.asarray([w1(pops[a][s], pops[b][s]) for s in seeds])
    mean = float(vals.mean())
    if vals.size >= 3:
        h = T95_DF2 * float(vals.std(ddof=1)) / np.sqrt(float(vals.size))
        lo, hi, kind = mean - h, mean + h, "95% Student-t CI"
    elif vals.size == 2:
        lo, hi, kind = float(vals.min()), float(vals.max()), "observed range"
    elif vals.size == 1:
        lo, hi, kind = mean, mean, "single seed (no interval)"
    else:
        raise ValueError(f"{a}-{b}: no paired seed")
    return {"mean_w1": mean, "interval_low": lo, "interval_high": hi,
            "interval_type": kind, "n_paired_seeds": int(vals.size),
            "paired_seeds": ";".join(str(s) for s in seeds)}


def paired_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def w1(a, b):
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def describe_fit(x, y):
    """Descriptive least-squares line. No inference is attached."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root",
                    default=str(REPO / "notes" / "pofd" / "cluster"))
    ap.add_argument("--pairwise-csv",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_equilibria"
                               / "model_pairwise_w1_round30.csv"))
    ap.add_argument("--cells-csv",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_model_equilibria"
                               / "model_equilibrium_cells.csv"))
    ap.add_argument("--mistral-seeds", default="42",
                    help="comma-separated Mistral seeds treated as valid "
                         "(default 42: seeds 0 and 43 are "
                         "parser-contaminated -- see the module docstring)")
    ap.add_argument("--out-dir",
                    default=str(REPO / "notes" / "pofd"
                               / "section3_map_vs_outcome"))
    args = ap.parse_args()
    mistral_seeds = tuple(int(x) for x in args.mistral_seeds.split(","))
    root = Path(args.runs_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zs, errs = load_zero_shot(root)
    missing = [m for m in MODELS if m not in zs]
    if missing:
        sys.exit(f"[mvo] no zero-shot map for {missing}")
    dep = load_deployed(Path(args.cells_csv), root)

    # RECOMPUTED y axis. Verify the recomputation against the published
    # artifact on the seeds that artifact used, so the only difference
    # between them is the deliberate Mistral exclusion.
    pub = {}
    with Path(args.pairwise_csv).open(newline="") as fh:
        for r in csv.DictReader(fh):
            pub[frozenset((r["model_a"], r["model_b"]))] = r
    if len(pub) != 15:
        sys.exit(f"[mvo] expected 15 outcome pairs, got {len(pub)}")
    pops_pub = load_final_populations(Path(args.cells_csv), root, (42, 43))
    for a, b in combinations(MODELS, 2):
        got = outcome_row(a, b, pops_pub)["mean_w1"]
        want = float(pub[frozenset((a, b))]["mean_w1"])
        if abs(got - want) > 1e-12:
            errs.append(f"{a}-{b}: recomputed W1 {got:.12f} != published "
                        f"{want:.12f} -- the y axis is not the same "
                        f"quantity the figure reports")
    pops = (pops_pub if set(mistral_seeds) == {42, 43}
            else load_final_populations(Path(args.cells_csv), root,
                                        mistral_seeds))

    recs = []
    for a, b in combinations(MODELS, 2):
        yr = outcome_row(a, b, pops)
        seeds = [int(s) for s in yr["paired_seeds"].split(";")]
        dmae = [paired_mae(dep[a][s], dep[b][s])
                for s in seeds if s in dep[a] and s in dep[b]]
        recs.append({
            "model_a": a, "model_b": b,
            "label_a": LABEL[a], "label_b": LABEL[b],
            "pair": f"{SHORT[a]}-{SHORT[b]}",
            "within_family": FAMILY[a] == FAMILY[b],
            "zeroshot_map_mae": paired_mae(zs[a], zs[b]),
            "zeroshot_map_w1": w1(zs[a], zs[b]),
            "deployed_map_mae": float(np.mean(dmae)) if dmae else float("nan"),
            "n_deployed_seeds": len(dmae),
            "final_pop_w1": yr["mean_w1"],
            "final_pop_w1_low": yr["interval_low"],
            "final_pop_w1_high": yr["interval_high"],
            "n_paired_seeds": yr["n_paired_seeds"],
            "paired_seeds": yr["paired_seeds"],
            "interval_type": yr["interval_type"],
        })
    recs.sort(key=lambda r: r["zeroshot_map_mae"])

    x = [r["zeroshot_map_mae"] for r in recs]
    y = [r["final_pop_w1"] for r in recs]
    rho, _ = spearmanr(x, y)
    fit = describe_fit(x, y)
    alt = {
        "zeroshot_w1": float(spearmanr([r["zeroshot_map_w1"] for r in recs],
                                       y).statistic),
        "deployed_mae": float(spearmanr(
            [r["deployed_map_mae"] for r in recs], y).statistic),
    }
    # THE SAME CORRELATION UNDER THE OTHER MISTRAL SEED SET, so the
    # reader can see whether excluding the contaminated cell matters.
    other = (42, 43) if set(mistral_seeds) == {42} else (42,)
    pops_o = load_final_populations(Path(args.cells_csv), root, other)
    y_o = [outcome_row(a, b, pops_o)["mean_w1"]
           for a, b in combinations(MODELS, 2)]
    x_o = [paired_mae(zs[a], zs[b]) for a, b in combinations(MODELS, 2)]
    mistral_sensitivity = {
        "used": list(mistral_seeds),
        "used_spearman_rho": float(rho),
        "alternative": list(other),
        "alternative_spearman_rho": float(spearmanr(x_o, y_o).statistic),
    }
    # leave-one-model-out: each model sits in 5 of the 15 pairs
    loo = {}
    for m in MODELS:
        keep = [r for r in recs
                if r["model_a"] != m and r["model_b"] != m]
        loo[m] = {
            "n_pairs": len(keep),
            "spearman_rho": float(spearmanr(
                [r["zeroshot_map_mae"] for r in keep],
                [r["final_pop_w1"] for r in keep]).statistic),
            "slope": describe_fit([r["zeroshot_map_mae"] for r in keep],
                                  [r["final_pop_w1"] for r in keep])["slope"],
        }

    csv_p = out / "section3_map_vs_outcome.csv"
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- figure ---------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    xs = np.linspace(min(x) * .92, max(x) * 1.04, 50)
    ax.plot(xs, fit["slope"] * xs + fit["intercept"], color="#9aa0a6",
            lw=1.1, ls=(0, (4, 2.5)), zorder=1)
    for r in recs:
        c = WITHIN_C if r["within_family"] else ACROSS_C
        ax.plot([r["zeroshot_map_mae"]] * 2,
                [r["final_pop_w1_low"], r["final_pop_w1_high"]],
                color=c, lw=1.2, alpha=.85, solid_capstyle="round",
                zorder=2)
        mk = {1: "^", 2: "s"}.get(r["n_paired_seeds"], "o")
        ax.plot([r["zeroshot_map_mae"]], [r["final_pop_w1"]], mk, ms=4.8,
                color=c, mec="white", mew=.7, zorder=3)
        ax.annotate(r["pair"], (r["zeroshot_map_mae"], r["final_pop_w1"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=5.9, color="#4a4e53")
    ax.set_xlabel("Initial zero-shot prediction-map distance\n"
                  "(mean $|p_A(i)-p_B(i)|$ over matched agents)", labelpad=4)
    ax.set_ylabel("Final population distance\n"
                  "($W_1$, round 30, paired seeds)", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(.03, .95,
            f"Spearman $\\rho$ = {rho:.2f}  (15 pairs, not independent)\n"
            f"descriptive fit: slope {fit['slope']:.2f}, "
            f"$R^2$ {fit['r2']:.2f}",
            transform=ax.transAxes, va="top", fontsize=7,
            color="#3a3e43")
    handles = [
        Line2D([], [], color=WITHIN_C, marker="o", ms=4.8, lw=1.2,
               label="within family"),
        Line2D([], [], color=ACROSS_C, marker="o", ms=4.8, lw=1.2,
               label="across family"),
    ]
    ns = {r["n_paired_seeds"] for r in recs}
    if 2 in ns:
        handles.append(Line2D([], [], color="#7a7e83", marker="s", ms=4.6,
                              lw=0, label="2 paired seeds "
                                          "(range, not CI)"))
    if 1 in ns:
        handles.append(Line2D([], [], color="#7a7e83", marker="^", ms=4.8,
                              lw=0, label="1 valid seed (no interval)"))
    ax.legend(handles=handles, frameon=False, fontsize=6.6,
              loc="lower right", handlelength=1.5)
    fig.tight_layout()
    pdf, png = (out / "section3_map_vs_outcome.pdf",
                out / "section3_map_vs_outcome.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    payload = {
        "question": "is the spread of Figure 3(a)'s final populations "
                    "inherited from how differently the six base models "
                    "predict to begin with?",
        "x_axis": {
            "primary": "agent-paired mean |p_A(i)-p_B(i)| between the two "
                       "models' round-0 FROZEN prediction maps",
            "source": "pofds3i_<model>_d8_greedy_..._r30 pred_raw[0] "
                      "(personal-history ICL wave, weights never trained)",
            "seed_invariance": "verified bit-identical across seeds "
                               "{0,42,43} for all six models",
            "secondary": "equal-mass W1 between the same two maps",
            "alternate": "round-0 map of the Figure-3(a) SFT runs "
                         "themselves (post round-0 SFT, seed-dependent, "
                         "averaged over paired seeds)",
        },
        "y_axis": "paired-seed equal-mass W1 between round-30 post-peer "
                  "populations, from analyze_section3_model_pairwise_w1.py",
        "independence": (
            "the 15 pairs are NOT independent -- each of the 6 models "
            "appears in 5 of them. Spearman's rho and the fitted line are "
            "descriptive summaries; no p-value and no confidence band is "
            "reported, and the leave-one-model-out sweep below is the "
            "sensitivity offered in their place."),
        "mistral": {
            "valid_strict_seeds_used": list(mistral_seeds),
            "gate_2026_08_28": {
                "seed_0": "FAIL -- round 1 parse_fail_frac 1.0: all 723 "
                          "generations malformed, every agent served the "
                          "parser 0.5 default; final population mean "
                          "0.491383",
                "seed_42": "PASS -- clean in every round; final "
                           "population mean 0.540000",
                "seed_43": "FAIL -- round 25 parse_fail_frac 1.0 (all 723 "
                           "generations '58 (58'); from round 26 the "
                           "model emits '0.50 (' with parse_fail_frac 0, "
                           "carrying the injected default forward. Final "
                           "population mean 0.500000, and the failure is "
                           "INSIDE the late window.",
            },
            "consequence": (
                "Mistral's actual valid seed count is ONE. Its five pairs "
                "are a single observation with no interval -- not a "
                "two-sample range, and never a two-sample CI. The "
                "published model_pairwise_w1_round30.csv uses seeds "
                "{42,43}; this analysis reproduces that exactly on those "
                "seeds (asserted) and then excludes seed 43."),
            "deployed_round0_map": (
                "degenerate: constant 1.0 for all 723 agents, which is "
                "why the deployed-map x axis is reported only as an "
                "alternate"),
        },
        "mistral_seed_sensitivity": mistral_sensitivity,
        "spearman_rho_primary": float(rho),
        "spearman_rho_alternates": alt,
        "descriptive_fit": fit,
        "leave_one_model_out": loo,
        "zero_shot_map_summary": {
            m: {"mean": float(zs[m].mean()), "sd": float(zs[m].std()),
                "n_distinct": int(np.unique(zs[m]).size)} for m in MODELS},
        "gate": {"errors": errs, "pass": not errs},
        "pairs": recs,
    }
    (out / "section3_map_vs_outcome.json").write_text(
        json.dumps(payload, indent=2))

    cap = [
        "Initial prediction-map distance versus final population",
        "distance, over all 15 pairs of the six Figure-3(a) models. x:",
        "how far apart two models' round-0 ZERO-SHOT prediction maps are,",
        "as the mean absolute difference over the same 723 MovieLens/",
        "Action agents; taken from the frozen personal-history ICL runs,",
        "whose weights never train, and verified bit-identical across",
        "seeds, so x carries no seed uncertainty. y: 1-Wasserstein",
        "distance between the two models' round-30 post-peer populations,",
        "paired within seed. Bars are across-seed 95% Student-t intervals",
        "on three paired seeds (circles). Mistral is strict-parse valid",
        "at ONE seed only -- seeds 0 and 43 each lose a whole round to",
        "malformed generations that served the parser's 0.5 default, and",
        "on seed 43 that round is inside the late window -- so its five",
        "pairs are single observations with no interval (triangles).",
        "Blue: two models of the same family. The",
        f"dashed line is a descriptive least-squares fit (slope",
        f"{fit['slope']:.2f}, R2 {fit['r2']:.2f}) and rho is Spearman's",
        "rank correlation. NEITHER carries inference: each model appears",
        "in five pairs, so the 15 points are not independent observations,",
        "and no p-value or confidence band is reported for them.",
    ]
    (out / "section3_map_vs_outcome_caption.txt").write_text(
        "\n".join(cap) + "\n")

    if errs:
        print("[mvo] GATE FAIL:")
        for e in errs:
            print("   -", e)
    else:
        print("[mvo] GATE PASS: six frozen zero-shot maps, "
              "seed-invariant, 723 agents")
    hdr = (f"{'pair':<10}{'fam':<8}{'zs_map_mae':>11}{'zs_map_w1':>11}"
           f"{'dep_mae':>9}{'final_W1':>10}{'n':>3} {'interval':>16}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['pair']:<10}"
              f"{'within' if r['within_family'] else 'across':<8}"
              f"{r['zeroshot_map_mae']:>11.4f}{r['zeroshot_map_w1']:>11.4f}"
              f"{r['deployed_map_mae']:>9.4f}{r['final_pop_w1']:>10.4f}"
              f"{r['n_paired_seeds']:>3} {r['interval_type']:>16}")
    print(f"\nSpearman rho (zero-shot MAE vs final W1) = {rho:+.4f}   "
          f"[15 pairs, NOT independent -- no p-value]")
    print(f"  alternate x = zero-shot W1     rho = {alt['zeroshot_w1']:+.4f}")
    print(f"  alternate x = deployed r0 MAE  rho = "
          f"{alt['deployed_mae']:+.4f}")
    print(f"  Mistral seeds {mistral_sensitivity['used']} -> rho "
          f"{mistral_sensitivity['used_spearman_rho']:+.4f}   |   seeds "
          f"{mistral_sensitivity['alternative']} -> rho "
          f"{mistral_sensitivity['alternative_spearman_rho']:+.4f}")
    print(f"descriptive fit: y = {fit['slope']:.3f} x + "
          f"{fit['intercept']:.3f}   R2 = {fit['r2']:.3f}")
    print("leave-one-model-out rho:")
    for m, v in loo.items():
        print(f"  drop {LABEL[m]:<10} n={v['n_pairs']:>2}  "
              f"rho={v['spearman_rho']:+.4f}  slope={v['slope']:+.3f}")
    print(f"\n[mvo] wrote {out}/section3_map_vs_outcome"
          f".{{csv,json,pdf,png}} + _caption.txt")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
