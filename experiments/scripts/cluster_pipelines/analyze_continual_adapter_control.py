#!/usr/bin/env python3
"""CONTINUAL-ADAPTER CONTROL: fresh-each-round vs continual LoRA weights.

Reviewer question: the paper's SFT loops reset the adapter to its
pristine state before every round (FRESH_EACH_ROUND=1), which rules out
weight-drift compounding.  Does the result survive when the weights are
instead allowed to persist and keep training round after round?

WHAT "CONTINUAL" MEANS HERE, PRECISELY.  With FRESH_EACH_ROUND=0 the
LoRA ADAPTER WEIGHTS PERSIST ACROSS ROUNDS: no pristine snapshot is
taken and none is restored, so round t starts from the weights round
t-1 finished with (run_pokec_gated_lm.py:2276-2280, 2926-2927; the
learner's reset() is a documented no-op).  THE OPTIMIZER IS RECREATED
EACH ROUND: SFTLearner.train() builds a NEW SFTConfig and a NEW
SFTTrainer per call and passes no ``optimizers=`` argument, so HF's
Trainer constructs a fresh AdamW and a fresh LR schedule every round
(perfsim/learners/lm/sft.py:115-175, 236-253).  This is WEIGHT
carryover.  It is NOT optimizer-moment carryover -- Adam's first and
second moments start at zero in every round of both arms, and no
experiment in this repository carries them over.

THE PAIRING.  Both arms are the env3 forward-KL peer environment on
MovieLens/Action, 723 agents, 30 rounds, W=0.5, k=0.2, eps_ai=0.4,
eps_social=0.2, replace/723, LoRA r512, lr 5e-5, 1 epoch, batch 4,
seeds {0,42,43}.  The two HTCondor environments are byte-identical
except FRESH_EACH_ROUND (0 vs 1) and the W&B run suffix.

  continual  pofdws2fc_qwen7b_b{L}_ea0p4_w0p5_l0p2_es0p2_s{S}_fresh_data
  fresh      pofdws2f_ (same tag body)          -- forward-KL family
             pofdws2_  at lambda=0 seed 0 only  -- see below

DIRECTION-FREE REUSE AT lambda=0.  At kl_beta=0 the training style is
plain ``sft``: no KL term is computed, so KL_DIRECTION is inert.  The
lambda=0 seed-0 fresh cell therefore comes from the pre-forward ws2
wave (``pofdws2_``), whose submit environment differs from the forward
one ONLY by the absent KL_DIRECTION variable.  Seeds 42 and 43 use the
forward-family ``pofdws2f_`` cells.  This is the reuse convention
check_pofd_sanity already encodes ("b0 rows are direction-free").

OPERATOR CAVEAT -- READ THIS BEFORE QUOTING THE NUMBERS.  Every run in
BOTH arms records population_update = "nested_ai_then_social_v1", the
PRE-CORRECTION operator, with AB_SWEEPS=1.  No continual-weight run
exists under the corrected "nested_ai_anchored_then_social_v2" (anch2)
operator anywhere in the archive.  The contrast is therefore internally
matched -- both arms share one operator, so the fresh-vs-continual
difference is attributable to the weight protocol alone -- but it is
NOT a corrected-operator result, and the analysis says so in every
artifact it writes.

  python analyze_continual_adapter_control.py
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "perfsim-cac"))

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

SEEDS = (0, 42, 43)
# --wave fec    the archived PRE-CORRECTION pair (v1 operator)
# --wave anch2  the 2026-08-29 corrected repeat, both arms newly run
WAVES = {
    "fec": dict(
        lambdas=(("b0", 0.0), ("b0p5", 0.5), ("b1", 1.0)),
        body="qwen7b_{lam}_ea0p4_w0p5_l0p2_es0p2_s{seed}_fresh_data",
        cont_prefix="pofdws2fc_", fresh_prefix="pofdws2f_",
        # lambda=0 seed 0 has no forward-family twin: the direction-free
        # ws2 cell is the matched fresh run (see the docstring)
        fresh_override={("b0", 0): "pofdws2_"},
        operator="nested_ai_then_social_v1",
        raw_gen=False),
    "anch2": dict(
        lambdas=(("b0", 0.0), ("b2", 2.0)),
        body=("qwen7b_{lam}_ea0p4_w0p5_l0p2_es0p2_anch2_{arm}"
              "_s{seed}_r30"),
        cont_prefix="pofdcac_", fresh_prefix="pofdcac_",
        fresh_override={}, arm_tokens=("adfresh", "adcont"),
        operator="nested_ai_anchored_then_social_v2",
        raw_gen=True),
}

LATE = 5                     # final-five-round window (repo convention)
T95_DF2 = 4.302652729911275  # two-sided 95% Student-t, 2 dof

# every dial both arms must agree on; FRESH_EACH_ROUND is the ONE that
# is allowed -- required -- to differ.
MATCHED_KEYS = (
    "kl_beta", "training_style", "population_update", "anchor_mode",
    "eps", "eps_ai", "w_plat", "innate_lambda", "gamma_bias", "ab_sweeps",
    "pop_order", "n_rounds", "seed", "base_model", "dataset", "ml_target",
    "data_regime", "train_cap", "n_labeled", "lora_r", "sft_lr",
    "sft_epochs", "sft_batch_size", "seed_base_data", "pristine_frac",
    "icl_k", "icl_days", "do_sample", "deploy_every", "run_mode",
    "platform_sus_scale", "profile_drop_cols", "profile_permute_cols",
)
CORRECTED_OPERATOR = "nested_ai_anchored_then_social_v2"

CONT_C = "#c44e52"
FRESH_C = "#4c72b0"
INK = "#202328"


def run_dir(root: Path, w: dict, arm: str, lam: str, seed: int) -> Path:
    pre = (w["cont_prefix"] if arm == "continual" else
           w["fresh_override"].get((lam, seed), w["fresh_prefix"]))
    kw = {"lam": lam, "seed": seed}
    if "{arm}" in w["body"]:
        kw["arm"] = w["arm_tokens"][0 if arm == "fresh" else 1]
    return root / (pre + w["body"].format(**kw))


def load(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    tr = torch.load(d / "trajectory.pt", map_location="cpu",
                    weights_only=False)
    return cfg, tr


def stats(op: np.ndarray) -> dict:
    """op is [T, n] post-peer opinions. Repo conventions throughout:
    the late window is the final five rounds and drift is the half-split
    (mean of the last floor(w/2) rounds minus the first floor(w/2)),
    identical to analyze_section3_model_equilibria._stochastic_stats."""
    means = op.mean(axis=1)
    tail = means[-LATE:]
    half = LATE // 2
    return {
        "final_mean": float(means[-1]),
        "late_mean": float(tail.mean()),
        "late_min": float(tail.min()),
        "late_max": float(tail.max()),
        "late_range": float(tail.max() - tail.min()),
        "drift": float(tail[-half:].mean() - tail[:half].mean()),
        "pop_sd_final": float(op[-1].std(ddof=0)),
        "pop_sd_late": float(np.mean(op[-LATE:].std(axis=1, ddof=0))),
        "n_rounds": int(op.shape[0]),
        "n_agents": int(op.shape[1]),
    }


def w1_equal_mass(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein between two equal-size empirical populations."""
    if a.shape != b.shape:
        raise ValueError(f"population size mismatch {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def ci95(vals) -> tuple[float, float, float, float]:
    v = np.asarray(vals, dtype=float)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size == 3:
        h = T95_DF2 * sd / np.sqrt(3.0)
        return m, sd, m - h, m + h
    return m, sd, float(v.min()), float(v.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root",
                    default=str(REPO / "notes" / "pofd" / "cluster"))
    ap.add_argument("--wave", default="fec", choices=tuple(WAVES),
                    help="fec = the archived pre-correction pair; "
                         "anch2 = the corrected repeat")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    W = WAVES[args.wave]
    LAMBDAS = W["lambdas"]
    root = Path(args.runs_root)
    out = Path(args.out_dir) if args.out_dir else (
        REPO / "notes" / "pofd" / f"continual_adapter_control_{args.wave}"
        if args.wave != "fec" else
        REPO / "notes" / "pofd" / "continual_adapter_control")
    out.mkdir(parents=True, exist_ok=True)

    cells, gate_errs, gate_warns = [], [], []
    pops: dict[tuple[str, str, int], np.ndarray] = {}
    cfgs: dict[tuple[str, str, int], dict] = {}

    for lam_tok, lam in LAMBDAS:
        for seed in SEEDS:
            for arm in ("fresh", "continual"):
                d = run_dir(root, W, arm, lam_tok, seed)
                if not (d / "trajectory.pt").exists():
                    gate_errs.append(f"MISSING {arm} lam={lam} s={seed}: {d}")
                    continue
                cfg, tr = load(d)
                op = tr["op_raw"].float().numpy()
                pred = tr["pred_raw"].float().numpy()
                key = (arm, lam_tok, seed)
                pops[key] = op[-1].copy()
                cfgs[key] = cfg

                # ---- gates ------------------------------------------
                want_fresh = (arm == "fresh")
                if bool(cfg.get("fresh_each_round")) is not want_fresh:
                    gate_errs.append(
                        f"{d.name}: fresh_each_round="
                        f"{cfg.get('fresh_each_round')} but arm={arm}")
                if abs(float(cfg.get("kl_beta", -1)) - lam) > 1e-12:
                    gate_errs.append(f"{d.name}: kl_beta="
                                     f"{cfg.get('kl_beta')} != {lam}")
                if int(cfg.get("seed", -1)) != seed:
                    gate_errs.append(f"{d.name}: seed={cfg.get('seed')}")
                if op.shape[0] != int(cfg.get("n_rounds", 0)):
                    gate_errs.append(f"{d.name}: {op.shape[0]} rounds of "
                                     f"{cfg.get('n_rounds')}")
                if not np.isfinite(op).all() or not np.isfinite(pred).all():
                    gate_errs.append(f"{d.name}: non-finite op/pred")
                if pred.min() < 0.0 or pred.max() > 1.0:
                    gate_errs.append(f"{d.name}: pred outside [0,1] "
                                     f"[{pred.min():.4f},{pred.max():.4f}]")
                # DIGIT-FREE PROXY. This wave predates SAVE_RAW_GEN, so no
                # raw generation log exists and a direct malformed count is
                # impossible. HFCausalLMModel._parse returns exactly 0.5
                # when a generation holds no digit, so the share of served
                # values that are bit-exactly 0.5 upper-bounds the
                # malformed rate (a genuine 0.5 answer also lands here).
                default_frac = float(np.mean(pred == 0.5))
                if lam == 0.0 and float(cfg.get("kl_beta", 0)) == 0.0 and \
                        cfg.get("training_style") != "sft":
                    gate_errs.append(f"{d.name}: lam=0 but style="
                                     f"{cfg.get('training_style')}")
                if lam > 0 and cfg.get("kl_direction") != "forward":
                    gate_errs.append(f"{d.name}: kl_direction="
                                     f"{cfg.get('kl_direction')!r}")
                if cfg.get("population_update") != W["operator"]:
                    msg = (f"{d.name}: population_update="
                           f"{cfg.get('population_update')!r} (want "
                           f"{W['operator']!r})")
                    (gate_errs if args.wave == "anch2"
                     else gate_warns).append(msg)
                if args.wave == "anch2" \
                        and cfg.get("ai_gate_mode") == "all_open":
                    gate_errs.append(
                        f"{d.name}: ai_gate_mode=all_open -- anch2 is "
                        f"numerically identical to the legacy operator "
                        f"there, so the corrected control is void")
                if default_frac > 0.0:
                    gate_warns.append(f"{d.name}: {default_frac:.4%} of "
                                      f"served values are exactly 0.5")

                s = stats(op)
                cells.append(dict(
                    arm=arm, lam=lam, lam_token=lam_tok, seed=seed,
                    tag=d.name, kl_direction=cfg.get("kl_direction"),
                    training_style=cfg.get("training_style"),
                    population_update=cfg.get("population_update"),
                    fresh_each_round=bool(cfg.get("fresh_each_round")),
                    served_default_frac=f"{default_frac:.8f}",
                    **{k: (f"{v:.8f}" if isinstance(v, float) else v)
                       for k, v in s.items()}))

    # ---- paired matching audit ------------------------------------
    for lam_tok, lam in LAMBDAS:
        for seed in SEEDS:
            a = cfgs.get(("fresh", lam_tok, seed))
            b = cfgs.get(("continual", lam_tok, seed))
            if a is None or b is None:
                continue
            for k in MATCHED_KEYS:
                if a.get(k) != b.get(k):
                    gate_errs.append(f"lam={lam} s={seed}: {k} differs "
                                     f"(fresh={a.get(k)!r} "
                                     f"continual={b.get(k)!r})")
            # kl_direction is only meaningful when a KL term exists
            if lam > 0 and a.get("kl_direction") != b.get("kl_direction"):
                gate_errs.append(f"lam={lam} s={seed}: kl_direction differs")

    if not cells:
        print(f"[cac] no cells loaded for --wave {args.wave}; expected:")
        for e in gate_errs:
            print("   -", e)
        return 1
    cells_p = out / "continual_adapter_control_cells.csv"
    with cells_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)

    # ---- per-lambda summary ---------------------------------------
    def get(arm, lam_tok, field):
        return [float(c[field]) for c in cells
                if c["arm"] == arm and c["lam_token"] == lam_tok]

    rows = []
    for lam_tok, lam in LAMBDAS:
        rec = {"lam": lam, "lam_token": lam_tok, "n_seeds": len(SEEDS)}
        for field in ("late_mean", "pop_sd_final", "pop_sd_late", "drift",
                      "final_mean"):
            for arm in ("fresh", "continual"):
                m, sd, lo, hi = ci95(get(arm, lam_tok, field))
                rec[f"{arm}_{field}"] = m
                rec[f"{arm}_{field}_sd"] = sd
                if field in ("late_mean", "drift"):
                    rec[f"{arm}_{field}_lo"] = lo
                    rec[f"{arm}_{field}_hi"] = hi
            rec[f"delta_{field}"] = (rec[f"continual_{field}"]
                                     - rec[f"fresh_{field}"])
        w1s = [w1_equal_mass(pops[("fresh", lam_tok, s)],
                             pops[("continual", lam_tok, s)])
               for s in SEEDS
               if ("fresh", lam_tok, s) in pops
               and ("continual", lam_tok, s) in pops]
        m, sd, lo, hi = ci95(w1s)
        rec.update(w1_mean=m, w1_sd=sd, w1_lo=lo, w1_hi=hi,
                   w1_per_seed=";".join(f"{v:.8f}" for v in w1s))
        # scale reference: how far apart are two FRESH seeds of the same
        # cell?  A fresh-vs-continual W1 below this is inside seed noise.
        fw = [w1_equal_mass(pops[("fresh", lam_tok, a)],
                            pops[("fresh", lam_tok, b)])
              for a, b in ((0, 42), (0, 43), (42, 43))]
        rec["w1_fresh_seed_pairs_mean"] = float(np.mean(fw))
        rec["w1_fresh_seed_pairs_max"] = float(np.max(fw))
        rows.append(rec)

    sum_p = out / "continual_adapter_control.csv"
    with sum_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- figure ----------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .8,
                         "xtick.major.width": .8, "ytick.major.width": .8,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    x = np.arange(len(LAMBDAS), dtype=float)
    xl = [f"$\\lambda={l:g}$" for _, l in LAMBDAS]
    off = .15

    ax = axes[0]
    for i, r in enumerate(rows):
        ax.plot([x[i] - off, x[i] + off],
                [r["fresh_late_mean"], r["continual_late_mean"]],
                color="0.75", lw=1.0, zorder=2)
        for dx, arm, c in ((-off, "fresh", FRESH_C),
                           (off, "continual", CONT_C)):
            ax.plot([x[i] + dx] * 2,
                    [r[f"{arm}_late_mean_lo"], r[f"{arm}_late_mean_hi"]],
                    color=c, lw=1.3, solid_capstyle="round", zorder=3)
            ax.plot([x[i] + dx], [r[f"{arm}_late_mean"]], "o", ms=4.6,
                    color=c, mec="white", mew=.7, zorder=4)
    ax.set_ylabel("Final-five-round\npopulation mean", labelpad=3)

    ax = axes[1]
    for i, r in enumerate(rows):
        ax.plot([x[i] - off, x[i] + off],
                [r["fresh_pop_sd_late"], r["continual_pop_sd_late"]],
                color="0.75", lw=1.0, zorder=2)
        for dx, arm, c in ((-off, "fresh", FRESH_C),
                           (off, "continual", CONT_C)):
            ax.plot([x[i] + dx], [r[f"{arm}_pop_sd_late"]], "o", ms=4.6,
                    color=c, mec="white", mew=.7, zorder=4)
    ax.set_ylabel("Population SD\n(late window)", labelpad=3)

    ax = axes[2]
    for i, r in enumerate(rows):
        ax.plot([x[i]] * 2, [r["w1_lo"], r["w1_hi"]], color="#55595f",
                lw=1.3, solid_capstyle="round", zorder=3)
        ax.plot([x[i]], [r["w1_mean"]], "D", ms=4.4, color="#55595f",
                mec="white", mew=.7, zorder=4)
        ax.plot([x[i] - .22, x[i] + .22],
                [r["w1_fresh_seed_pairs_max"]] * 2, color="#9aa0a6",
                lw=1.1, ls=(0, (3, 2)), zorder=2)
    ax.set_ylabel("$W_1$(fresh, continual)\nfinal populations", labelpad=3)
    ax.set_ylim(bottom=0)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xl)
        ax.set_xlim(-.55, len(LAMBDAS) - .45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color=FRESH_C, marker="o", ms=4.6, lw=1.3,
               label="fresh adapter each round"),
        Line2D([], [], color=CONT_C, marker="o", ms=4.6, lw=1.3,
               label="continual adapter (weights persist)"),
        Line2D([], [], color="#9aa0a6", lw=1.1, ls=(0, (3, 2)),
               label="max fresh-vs-fresh seed $W_1$ (noise floor)"),
    ], frameon=False, fontsize=6.8, loc="lower center", ncol=3,
        bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .085, 1, 1))
    pdf, png = (out / "continual_adapter_control.pdf",
                out / "continual_adapter_control.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    # ---- json ------------------------------------------------------
    payload = {
        "question": "does the SFT result survive when LoRA weights "
                    "persist across rounds instead of being reset?",
        "weight_protocol": {
            "continual": "adapter weights persist across rounds "
                         "(FRESH_EACH_ROUND=0): no pristine snapshot is "
                         "taken and none is restored",
            "fresh": "adapter is restored to its pristine snapshot "
                     "before every round (FRESH_EACH_ROUND=1)",
            "optimizer": "recreated each round in BOTH arms -- "
                         "SFTLearner.train() builds a new SFTTrainer per "
                         "call and passes no optimizers=, so AdamW and "
                         "its LR schedule are constructed fresh every "
                         "round. This is weight carryover, NOT "
                         "optimizer-moment carryover.",
            "code": ["run_pokec_gated_lm.py:2276-2280,2926-2927",
                     "perfsim/learners/lm/sft.py:115-175,236-253"],
        },
        "surface": {
            "dataset": "movielens/Action", "n_agents": 723,
            "n_rounds": 30, "w_plat": 0.5, "innate_lambda": 0.2,
            "eps_ai": 0.4, "eps_social": 0.2, "gamma_bias": 0.0,
            "ab_sweeps": 1, "kl_direction": "forward",
            "data_regime": "replace", "train_cap": 723,
            "lora_r": 512, "sft_lr": 5e-05, "seeds": list(SEEDS),
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
        },
        "operator_caveat": (
            "BOTH arms record population_update="
            "'nested_ai_then_social_v1' -- the PRE-CORRECTION operator. "
            "No continual-weight run exists under the corrected "
            f"'{CORRECTED_OPERATOR}' operator. The contrast is "
            "internally matched (one operator on both sides) but is not "
            "a corrected-operator result."),
        "malformed_generations": (
            "this wave predates SAVE_RAW_GEN, so no raw generation log "
            "exists and a direct malformed count is impossible. The "
            "reported served_default_frac is the share of served values "
            "bit-exactly 0.5, which upper-bounds the digit-free rate."),
        "late_window": LATE,
        "drift_definition": "half-split: mean(last 2 rounds) - "
                            "mean(first 2 rounds) of the 5-round window",
        "distributional_distance": "1-Wasserstein between the two arms' "
                                   "final post-peer populations, equal "
                                   "mass, per seed",
        "gate": {"n_cells": len(cells), "errors": gate_errs,
                 "warnings": sorted(set(gate_warns)),
                 "pass": not gate_errs},
        "by_lambda": rows,
    }
    (out / "continual_adapter_control.json").write_text(
        json.dumps(payload, indent=2))

    cap = [
        "Continual-adapter control. Fresh (blue) restores the LoRA",
        "adapter to its pristine state before every round; continual",
        "(red) lets the adapter persist and keep training. In BOTH arms",
        "the optimizer is recreated each round, so this is weight",
        "carryover and not optimizer-moment carryover. Left: mean",
        "population opinion over the final five rounds, with across-seed",
        "95% Student-t intervals over seeds {0,42,43}. Centre: across-",
        "agent population SD averaged over the same window. Right:",
        "1-Wasserstein distance between the two arms' final populations,",
        "paired within seed; the dashed line is the largest",
        "fresh-vs-fresh seed distance at that lambda, i.e. the seed-noise",
        "floor the paired distance must clear to mean anything. Env3",
        "forward-KL peer environment, Qwen2.5-7B, MovieLens/Action, 723",
        "agents, 30 rounds, W=0.5, k=0.2, eps_AI=0.4, eps_social=0.2. The",
        "two HTCondor environments are identical except FRESH_EACH_ROUND.",
        "CAVEAT: both arms run the pre-correction operator",
        "nested_ai_then_social_v1; no continual-weight run exists under",
        "the corrected anch2 operator.",
    ]
    (out / "continual_adapter_control_caption.txt").write_text(
        "\n".join(cap) + "\n")

    # ---- console ---------------------------------------------------
    print(f"[cac] {len(cells)} cells from {root}")
    if gate_errs:
        print("[cac] GATE FAIL:")
        for e in gate_errs:
            print("   -", e)
    else:
        print("[cac] GATE PASS: pairing, seeds, betas, rounds, "
              "finiteness, range")
    for wmsg in sorted(set(gate_warns)):
        print("[cac] WARN:", wmsg)
    hdr = (f"{'lam':>4} {'fresh_mean':>11} {'cont_mean':>11} {'delta':>9} "
           f"{'fresh_sd':>9} {'cont_sd':>9} {'fr_drift':>9} "
           f"{'co_drift':>9} {'W1':>9} {'noise':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['lam']:>4g} {r['fresh_late_mean']:>11.5f} "
              f"{r['continual_late_mean']:>11.5f} "
              f"{r['delta_late_mean']:>+9.5f} "
              f"{r['fresh_pop_sd_late']:>9.5f} "
              f"{r['continual_pop_sd_late']:>9.5f} "
              f"{r['fresh_drift']:>+9.5f} {r['continual_drift']:>+9.5f} "
              f"{r['w1_mean']:>9.5f} "
              f"{r['w1_fresh_seed_pairs_max']:>9.5f}")
    print(f"\n[cac] wrote {out}/continual_adapter_control"
          f".{{csv,json,pdf,png}} + _cells.csv + _caption.txt")
    return 1 if gate_errs else 0


if __name__ == "__main__":
    sys.exit(main())
