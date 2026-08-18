#!/usr/bin/env python3
"""Figure-2 family-prior analysis (2026-08-18, fig2_family_prior_*).

THE QUESTION: how large must the forward-KL coefficient be before each
checkpoint's CLOSED-LOOP population stays recognizably its own -- i.e.
lands closest in W1 to the frozen K=0 endpoint of the SAME checkpoint
rather than some other family's?

SELECTION RULE (one shared setting, never per-model): among the betas
with full 6-checkpoint coverage at (ea1, es0.05, seed 0), pick the
SMALLEST beta whose late populations (rounds 25-29) are closest in W1
to their OWN frozen k0 endpoint for >= 5 of 6 checkpoints. If no beta
qualifies, that is reported -- no per-model settings are ever chosen.
Betas 0 / 0.5 / 1 (the completed fam wave) are evaluated alongside the
scout betas 2 / 4 / 8: if a small beta already qualifies, the report
says so and no confirmation submission is needed.

SECONDARY METRICS per beta:
  - prior-to-population distance correlation: Pearson + Spearman between
    the 15 pairwise W1 distances among the six M0 priors and the 15
    pairwise W1 distances among the six late populations
  - centered-quantile shape retention: per checkpoint, the Pearson
    correlation between the mean-centered quantile functions (1..99%)
    of M0 and of the pooled late population; reported as the mean
  - rounds-25-29 stability: per checkpoint, the mean W1 between
    consecutive late rounds (small = the late window is a settled
    state, not a transient)

M0 SOURCES (completed runs, never re-run): pred_raw[0] of the seed-0
pofdreachbase_ probes (qwen7b / olmo7b / mistral7b) and of the
pofdzsprior_ probes (qwen3_8b / olmo3_7b / ministral8b). P0 is the
shared innate vector. Frozen endpoints are the completed fam k0 runs
at es0.05.

FIGURE (only when a beta is selected): six checkpoint panels, 2 rows x
3 columns -- columns are the Qwen / OLMo / Mistral families (row 1 =
the incumbent checkpoint, row 2 = the newer sibling). Each panel shows
the SHARED P0 (the perfect-prediction reference, kept as its own
separate curve), that checkpoint's M0, and its late population under
the selected beta. The late MODEL distribution (pred_raw) is
deliberately not shown. No title text (paper figure style).

Outputs (--out-dir, default notes/pofd/fam_beta_analysis/):
  fam_beta_selection.csv, fam_beta_matchmatrix.csv,
  fam_prior_panels.png / .pdf
"""
import argparse
import csv
import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "analyze_reach", os.path.join(HERE, "analyze_sft_icl_reach.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)

MODELS = ["qwen7b", "qwen3_8b", "olmo7b", "olmo3_7b", "mistral7b",
          "ministral8b"]
# panel grid: columns = families, row 1 incumbent, row 2 newer sibling
PANEL_GRID = [["qwen7b", "olmo7b", "mistral7b"],
              ["qwen3_8b", "olmo3_7b", "ministral8b"]]
PANEL_LABEL = {"qwen7b": "Qwen2.5-7B", "qwen3_8b": "Qwen3-8B",
               "olmo7b": "OLMo-2-7B", "olmo3_7b": "OLMo-3-7B",
               "mistral7b": "Mistral-7B", "ministral8b": "Ministral-8B"}
BETA_ARMS = [(0.0, "b0"), (0.5, "b0p5"), (1.0, "b1"),
             (2.0, "b2"), (4.0, "b4"), (8.0, "b8")]
ES_TOK = "es0p05"
LATE = range(25, 30)
PRIOR_TAG = {
    "qwen7b": "pofdreachbase_qwen7b_w0p5_l0p2_es0_s0",
    "olmo7b": "pofdreachbase_olmo7b_w0p5_l0p2_es0_s0",
    "mistral7b": "pofdreachbase_mistral7b_w0p5_l0p2_es0_s0",
    "qwen3_8b": "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0",
    "olmo3_7b": "pofdzsprior_olmo3_7b_w0p5_l0p2_es0_s0",
    "ministral8b": "pofdzsprior_ministral8b_w0p5_l0p2_es0_s0",
}


def fam_tag(model, arm):
    return f"pofdfam_{model}_{arm}_ea1_w0p5_l0p2_{ES_TOK}_s0"


def w1(a, b):
    """Exact 1-D Wasserstein between equal-weight samples."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    if a.shape == b.shape:
        return float(np.abs(a - b).mean())
    q = np.linspace(0.005, 0.995, 512)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def _load(roots, tag):
    rd = AN.find_run(roots, tag)
    if rd is None:
        return None
    return torch.load(os.path.join(rd, "trajectory.pt"),
                      map_location="cpu", weights_only=False)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / den) if den > 0 else float("nan")


def _spearman(x, y):
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return _pearson(rk(np.asarray(x)), rk(np.asarray(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        os.path.join(REPO, "runs", "pokec_gated_lm"),
        os.path.join(REPO, "notes", "pofd", "cluster")])
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "notes", "pofd", "fam_beta_analysis"))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # M0 priors + shared P0 + frozen k0 endpoints (all completed runs)
    prior, innate = {}, None
    for m in MODELS:
        d = _load(args.roots, PRIOR_TAG[m])
        assert d is not None, f"M0 prior run missing: {PRIOR_TAG[m]}"
        prior[m] = d["pred_raw"][0].float().numpy()
        if innate is None:
            innate = d["innate"].float().numpy()
    k0_late = {}
    for m in MODELS:
        d = _load(args.roots, fam_tag(m, "k0"))
        assert d is not None, f"frozen endpoint missing: {fam_tag(m, 'k0')}"
        k0_late[m] = d["op_raw"].float().numpy()[list(LATE)]

    # per-beta evaluation over the checkpoints with data present
    match_rows, sel_rows = [], []
    qualifying = []
    pops_by_beta = {}
    for beta, arm in BETA_ARMS:
        pops = {}
        for m in MODELS:
            d = _load(args.roots, fam_tag(m, arm))
            if d is not None:
                pops[m] = d["op_raw"].float().numpy()[list(LATE)]
        if len(pops) < len(MODELS):
            missing = [m for m in MODELS if m not in pops]
            print(f"[fam_beta] beta={beta:g}: INCOMPLETE "
                  f"({len(pops)}/6; missing {missing}) -- skipped")
            continue
        pops_by_beta[beta] = pops
        # 6x6 late W1 to every checkpoint's frozen endpoint
        n_correct, margins = 0, []
        for m in MODELS:
            dists = {c: float(np.mean([w1(pops[m][r], k0_late[c][r])
                                       for r in range(len(LATE))]))
                     for c in MODELS}
            best = min(dists, key=dists.get)
            own = dists[m]
            runner_up = min(v for c, v in dists.items() if c != best)
            n_correct += int(best == m)
            margins.append(runner_up - dists[best] if best == m
                           else dists[best] - own)
            match_rows.append({"beta": beta, "model": m,
                               "own_w1": own, "best": best,
                               "best_w1": dists[best],
                               "own_is_best": int(best == m),
                               **{f"w1_to_{c}": dists[c]
                                  for c in MODELS}})
        # prior-to-population pairwise-distance correlation (15 pairs)
        pairs = [(a, b) for i, a in enumerate(MODELS)
                 for b in MODELS[i + 1:]]
        d_prior = [w1(prior[a], prior[b]) for a, b in pairs]
        d_pop = [float(np.mean([w1(pops[a][r], pops[b][r])
                                for r in range(len(LATE))]))
                 for a, b in pairs]
        # centered-quantile shape retention M0 -> pooled late pop
        qgrid = np.linspace(0.01, 0.99, 99)
        rets = []
        for m in MODELS:
            qm = np.quantile(prior[m], qgrid)
            qp = np.quantile(pops[m].ravel(), qgrid)
            rets.append(_pearson(qm - qm.mean(), qp - qp.mean()))
        # late-window stability: mean consecutive-round W1
        stab = [float(np.mean([w1(pops[m][r], pops[m][r + 1])
                               for r in range(len(LATE) - 1)]))
                for m in MODELS]
        row = {"beta": beta, "arm": arm, "n_correct": n_correct,
               "qualifies": int(n_correct >= 5),
               "mean_own_w1": float(np.mean(
                   [r_["own_w1"] for r_ in match_rows
                    if r_["beta"] == beta])),
               "mean_margin": float(np.mean(margins)),
               "prior_pop_pearson": _pearson(d_prior, d_pop),
               "prior_pop_spearman": _spearman(d_prior, d_pop),
               "shape_retention_mean": float(np.mean(rets)),
               "shape_retention_min": float(np.min(rets)),
               "stability_w1_mean": float(np.mean(stab)),
               "stability_w1_max": float(np.max(stab))}
        sel_rows.append(row)
        if n_correct >= 5:
            qualifying.append(beta)
        print(f"[fam_beta] beta={beta:g}: own-match {n_correct}/6  "
              f"prior-corr r={row['prior_pop_pearson']:.3f} "
              f"rho={row['prior_pop_spearman']:.3f}  "
              f"shape {row['shape_retention_mean']:.3f}  "
              f"stability(W1) {row['stability_w1_mean']:.4f}")

    selected = min(qualifying) if qualifying else None
    if selected is None:
        print("\n[fam_beta] NO beta reaches 5/6 own-family matching -- "
              "no shared setting is selected (per-model settings are "
              "never chosen).")
    else:
        note = ("already covered by the completed wave -- no "
                "confirmation submission needed" if selected <= 1.0
                else f"submit fig2_family_prior_b{selected:g}_confirm")
        print(f"\n[fam_beta] SELECTED shared beta: {selected:g} "
              f"(smallest with >=5/6 own-family matching; {note})")

    def write(name, rows):
        if not rows:
            print(f"[fam_beta] {name}: no rows")
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, name), "w",
                  newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"[fam_beta] wrote {name} ({len(rows)} rows)")

    write("fam_beta_selection.csv", sel_rows)
    write("fam_beta_matchmatrix.csv", match_rows)

    if selected is None or args.no_fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arm_sel = dict((b, a) for b, a in BETA_ARMS)[selected]
    bins = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True,
                             sharey=True)
    for r_i, row_models in enumerate(PANEL_GRID):
        for c_i, m in enumerate(row_models):
            ax = axes[r_i][c_i]
            # P0: the shared perfect-prediction reference, its own curve
            ax.hist(innate, bins=bins, density=True, histtype="stepfilled",
                    color="0.85", edgecolor="0.6", lw=0.8,
                    label="$P^0$ (perfect prediction)")
            ax.hist(prior[m], bins=bins, density=True, histtype="step",
                    color="#1f77b4", ls="--", lw=1.6, label="$M^0$")
            ax.hist(pops_by_beta[selected][m].ravel(), bins=bins,
                    density=True, histtype="step", color="#d62728",
                    lw=1.6,
                    label=rf"late pop ($\beta={selected:g}$)")
            ax.text(0.03, 0.93, PANEL_LABEL[m], transform=ax.transAxes,
                    fontsize=9, va="top")
            if r_i == 1:
                ax.set_xlabel("opinion")
            if c_i == 0:
                ax.set_ylabel("density")
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.out_dir, f"fam_prior_panels.{ext}")
        fig.savefig(p, dpi=200)
        print(f"[fam_beta] wrote {os.path.basename(p)}")


if __name__ == "__main__":
    main()
