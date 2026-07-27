#!/usr/bin/env python3
"""Feature endogenization under the fresh-data protocol.

A protected attribute that carries NO incremental signal in the original data
can acquire one once the loop runs: the deployed model's own demographic
structure is written into the population's opinions, and the next round's
training data then contains it. The archived version of this analysis
(experiments/MMHD_restructured_project/) was measured on the OLD run family,
which accumulated LoRA weights across rounds (fresh_each_round=false), so the
rise there could always have been weight drift compounding. Every run used
here resets the adapter to its pristine snapshot before each round's training,
so nothing carries across rounds except the DATA the population generated.

Metric (identical to the archived one, so the numbers are comparable):
crossfit 5-fold OLS incremental R^2 of a demographic feature over the 10
MovieLens taste columns, fold seed 0. dR2 > 0 means the feature explains
variance in opinions that tastes alone do not.

Three panels, matched Qwen2.5-7B / MovieLens-Action runs at eps_AI=0.4:
  (a) population dR2_gender over rounds at beta=1, one line per population
      environment, against three zero references: the gender-label
      permutation null, the matched no-AI twin, and age as a placebo feature
  (b) served-model dR2_gender over rounds at beta=1, same environments
  (c) final-round population dR2_gender against beta, log(1+beta) spacing

Outputs (notes/pofd/figures/): feature_endogenization_qwen.{png,pdf},
feature_endogenization_qwen_points.csv. Caption printed.

Usage: python3 experiments/llm/plot_feature_endogenization.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_empirical_return_maps as erm            # noqa: E402  (rcParams too)
import plot_beta_dose_response_envs as env_mod      # noqa: E402  (env_cell)
import matplotlib.pyplot as plt                     # noqa: E402

OUT_DIR = erm.OUT_DIR
EPS_AI = env_mod.EPS_AI
FOCUS_BETA = 1.0        # only beta with a nonzero effect; panels (a)/(b)
N_PERM = 200            # gender-label shuffles for the null band
PERM_SEED = 7
NON_TASTE = ("age", "gender", "occ")

ENV_STYLE = {   # (eps_soc, W, lambda) -> (label, color, linestyle, marker)
    (0.0, 1.0, 0.0): ("full adoption", "#888888", (0, (3, 1, 1, 1)), "^"),
    (0.0, 0.5, 0.2): ("anchored", "#E69F00", "--", "s"),
    (0.2, 0.5, 0.2): ("anchored + peers", "#0072B2", "-", "o"),
}
C_NULL, C_TWIN, C_PLACEBO = "#cccccc", "#888888", "#009E73"


# ---------------------------------------------------------------- metric ----
def crossfit_r2(X, y, folds=5, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    yhat = np.full(n, np.nan)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
        Xte = np.column_stack([np.ones(len(te)), X[te]])
        b, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        yhat[te] = Xte @ b
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def incr_r2(tastes, extra, y):
    """Incremental R^2 of `extra` over `tastes` in predicting y. NaN when y is
    constant (a fully captured round has no variance left to explain)."""
    if np.std(y) < 1e-8:
        return np.nan
    return (crossfit_r2(np.column_stack([tastes, extra]), y)
            - crossfit_r2(tastes, y))


def features_of(run):
    """Taste matrix, gender indicator (M=1), age -- from the run's profiles."""
    prof = run["profiles"]
    keys = [k for k in prof if k not in NON_TASTE]
    assert "Drama" in prof and len(keys) == 10, \
        f"{run['tag']}: unexpected profile columns {keys}"
    tastes = np.column_stack([np.asarray(prof[k], np.float64) for k in keys])
    gender = (np.asarray(prof["gender"]) == "M").astype(np.float64)
    age = np.asarray(prof["age"], np.float64)
    return tastes, gender, age


def series(tastes, extra, arr2d):
    a = np.clip(np.asarray(arr2d, np.float64), 0.0, 1.0)
    return np.array([incr_r2(tastes, extra[:, None], a[t])
                     for t in range(a.shape[0])])


def perm_null(tastes, gender, y, n=N_PERM, seed=PERM_SEED):
    """Null distribution of dR2 with the gender labels shuffled (tastes and
    opinions untouched), i.e. what this estimator returns on a feature that
    carries no information about y."""
    rng = np.random.default_rng(seed)
    y = np.clip(np.asarray(y, np.float64), 0.0, 1.0)
    return np.array([incr_r2(tastes, rng.permutation(gender)[:, None], y)
                     for _ in range(n)])


# ------------------------------------------------------------------ main ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, candidates = erm.select_runs()

    envs = []       # (key, label, {beta: loaded run})
    for key, (label, *_rest) in ENV_STYLE.items():
        cell = env_mod.env_cell(candidates, *key)
        betas = sorted(cell)
        print(f"\n[env] {label}: betas {betas}")
        loaded = {}
        for b in betas:
            print(f"  beta={b:g}: {cell[b]['tag']}")
            loaded[b] = erm.load_run(cell[b])
        envs.append((key, label, loaded))
    beta_sets = {tuple(sorted(l)) for _, _, l in envs}
    assert len(beta_sets) == 1, f"beta grids differ across envs: {beta_sets}"
    betas = sorted(envs[0][2])
    assert FOCUS_BETA in betas, f"beta={FOCUS_BETA} missing from {betas}"

    # the profiles are one fixed population; verify every run shares it
    ref_t, ref_g, ref_a = features_of(envs[0][2][betas[0]])
    for _, label, loaded in envs:
        for b, r in loaded.items():
            t, g, a = features_of(r)
            assert np.array_equal(t, ref_t) and np.array_equal(g, ref_g) \
                and np.array_equal(a, ref_a), \
                f"{r['tag']}: profiles differ from the reference population"
    tastes, gender, age = ref_t, ref_g, ref_a
    n_m = int(gender.sum())
    print(f"\n[pop] n={len(gender)} (M={n_m}, F={len(gender) - n_m}); "
          f"10 taste columns; one fixed population across all runs")

    # ---- reference levels: original data, permutation null, twin, placebo ----
    innate = np.clip(envs[0][2][betas[0]]["innate"], 0.0, 1.0)
    dr2_innate = incr_r2(tastes, gender[:, None], innate)
    print(f"[ref] innate opinions: dR2_gender = {dr2_innate:+.4f} "
          f"(gender adds nothing to tastes in the ORIGINAL data)")

    nulls = []
    for key, label, loaded in envs:
        nulls.append(perm_null(tastes, gender, loaded[FOCUS_BETA]["op"][-1]))
    null_all = np.concatenate(nulls)
    null_lo = float(np.quantile(null_all, 0.05))
    null_hi = float(np.quantile(null_all, 0.95))
    print(f"[null] {N_PERM} gender-label shuffles x {len(envs)} envs at "
          f"beta={FOCUS_BETA:g}, final round: mean {null_all.mean():+.5f}, "
          f"5-95% [{null_lo:+.5f}, {null_hi:+.5f}], "
          f"max {null_all.max():+.5f}")

    # ---- per-env series ----
    stats = []
    for key, label, loaded in envs:
        r = loaded[FOCUS_BETA]
        s_op = series(tastes, gender, r["op"])
        s_pr = series(tastes, gender, r["pred"])
        s_age = series(tastes, age, r["op"])
        s_twin = (series(tastes, gender, r["twin"])
                  if r["twin"] is not None else None)
        finals = np.array([series(tastes, gender, loaded[b]["op"])[-1]
                           for b in betas])
        preds_f = np.array([series(tastes, gender, loaded[b]["pred"])[-1]
                            for b in betas])
        stats.append(dict(key=key, label=label, tag=r["tag"], op=s_op,
                          pred=s_pr, age=s_age, twin=s_twin, finals=finals,
                          preds_f=preds_f))
        mark = [0, 5, 10, 15, 20, 25, 29]
        print(f"\n[{label}] beta={FOCUS_BETA:g}  {r['tag']}")
        print("  round        " + " ".join(f"{t:7d}" for t in mark))
        print("  population   " + " ".join(f"{s_op[t]:+7.4f}" for t in mark))
        print("  served model " + " ".join(f"{s_pr[t]:+7.4f}" for t in mark))
        print("  age placebo  " + " ".join(f"{s_age[t]:+7.4f}" for t in mark))
        if s_twin is not None:
            print("  no-AI twin   " + " ".join(f"{s_twin[t]:+7.4f}"
                                               for t in mark))
        print("  final dR2_op by beta: "
              + ", ".join(f"b{b:g}={v:+.4f}" for b, v in zip(betas, finals)))

    # ---- verdicts stated from the numbers, not asserted ----
    print(f"\n[verdict] final-round population dR2_gender vs the null 95th "
          f"percentile ({null_hi:+.4f}):")
    for d in stats:
        v = d["op"][-1]
        rel = ("above" if v > null_hi else "below" if v < null_lo else "inside")
        mult = v / null_hi if null_hi > 0 else float("nan")
        print(f"  {d['label']:>16}: {v:+.4f} ({rel} the null band"
              + (f", {mult:.0f}x the 95th pct)" if v > null_hi else ")"))
    onsets = []
    for d in stats:
        on = next((b for b, v in zip(betas, d["finals"]) if v > null_hi), None)
        onsets.append(on)
        print(f"  {d['label']:>16}: first beta clearing the null band: "
              + (f"{on:g}" if on is not None else "none"))

    # ------------------------------------------------------------ figure ----
    fig, (a, bx, c) = plt.subplots(1, 3, figsize=(7.0, 2.6),
                                   layout="constrained")
    heads = ["gender predicts opinions only\nwhen the population is anchored",
             "capture erases the model's gender\nsignal; anchoring grows it",
             "endogenization needs more\nregularization than capture does"]
    for ax, pl, head in zip((a, bx, c), "abc", heads):
        ax.set_title(f"({pl})", loc="left", fontweight="bold", pad=4)
        ax.set_title(head, loc="right", fontsize=7.5, color="#666666", pad=4)
        ax.axhline(0, color="#aaaaaa", lw=0.7, zorder=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    rounds = np.arange(len(stats[0]["op"]))
    for ax, field in ((a, "op"), (bx, "pred")):
        ax.axhspan(null_lo, null_hi, color=C_NULL, zorder=0)
        ax.set_xlabel("round $t$")
        for d in stats:
            _, col, ls, mk = ENV_STYLE[d["key"]]
            ax.plot(rounds, d[field], color=col, ls=ls, lw=1.2,
                    label=d["label"])
    a.set_ylabel("population $\\Delta R^2$ of gender")
    bx.set_ylabel("served-model $\\Delta R^2$ of gender")

    # zero-references live in panel (a): twin, age placebo, permutation null
    twin_d = next((d for d in stats if d["twin"] is not None), None)
    if twin_d is not None:
        a.plot(rounds, twin_d["twin"], color=C_TWIN, ls=":", lw=1.0,
               label="no-AI twin")
    a.plot(rounds, stats[-1]["age"], color=C_PLACEBO, ls=":", lw=1.0,
           label="age (placebo)")
    a.legend(loc="upper left", frameon=False, fontsize=5.8)
    a.text(rounds[-1], null_hi, " null", fontsize=5.8, color="#8a8a8a",
           ha="right", va="bottom")
    bx.legend(loc="upper right", frameon=False, fontsize=5.8)

    xs = np.log1p(betas)
    c.axhspan(null_lo, null_hi, color=C_NULL, zorder=0)
    for d in stats:
        _, col, ls, mk = ENV_STYLE[d["key"]]
        c.plot(xs, d["finals"], color=col, ls=ls, lw=1.2, marker=mk, ms=3.2,
               label=d["label"])
    c.set_xticks(xs, [f"{b:g}" for b in betas])
    c.set_xlabel("$\\beta_{\\mathrm{KL}}$   ($\\log(1{+}\\beta)$ spacing)")
    c.set_ylabel("final-round population $\\Delta R^2$")
    c.legend(loc="upper left", frameon=False, fontsize=5.8)

    fig.text(0.995, 0.005, "single seed (s0)", fontsize=5.8, color="#999999",
             ha="right", va="bottom")

    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"feature_endogenization_qwen.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        outs.append(p)
    plt.close(fig)
    print(f"\n[fig] wrote {outs}")

    # --------------------------------------------------------------- CSV ----
    csv_path = os.path.join(OUT_DIR, "feature_endogenization_qwen_points.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "env", "run_tag", "beta", "round", "quantity",
                    "value"])
        for d in stats:
            for t in rounds:
                w.writerow(["a", d["label"], d["tag"], FOCUS_BETA, t,
                            "dR2_gender_population", f"{d['op'][t]:.6f}"])
                w.writerow(["a", d["label"], d["tag"], FOCUS_BETA, t,
                            "dR2_age_population_placebo", f"{d['age'][t]:.6f}"])
                if d["twin"] is not None:
                    w.writerow(["a", d["label"], d["tag"], FOCUS_BETA, t,
                                "dR2_gender_noAI_twin", f"{d['twin'][t]:.6f}"])
                w.writerow(["b", d["label"], d["tag"], FOCUS_BETA, t,
                            "dR2_gender_served_model", f"{d['pred'][t]:.6f}"])
        for d in stats:
            for b, v, pv in zip(betas, d["finals"], d["preds_f"]):
                w.writerow(["c", d["label"], "", b, 29,
                            "dR2_gender_population_final", f"{v:.6f}"])
                w.writerow(["c", d["label"], "", b, 29,
                            "dR2_gender_served_final", f"{pv:.6f}"])
        w.writerow(["ref", "", "", "", "", "dR2_gender_innate_opinions",
                    f"{dr2_innate:.6f}"])
        w.writerow(["ref", "", "", FOCUS_BETA, 29,
                    f"perm_null_p05_{N_PERM}x{len(envs)}", f"{null_lo:.6f}"])
        w.writerow(["ref", "", "", FOCUS_BETA, 29,
                    f"perm_null_p95_{N_PERM}x{len(envs)}", f"{null_hi:.6f}"])
    print(f"[csv] wrote {csv_path}")

    v_full, v_anch, v_peer = (d["op"][-1] for d in stats)
    p_peer = stats[-1]["pred"]
    print(f"""
[caption] feature_endogenization_qwen
Gender becomes predictive of opinions inside the loop, but only where the
population is anchored. Incremental R^2 of gender over the 10 MovieLens taste
columns (crossfit 5-fold OLS, fold seed 0), matched Qwen2.5-7B/MovieLens-Action
runs at eps_AI={EPS_AI:g}, 723 agents (M={n_m}), 30 rounds, replace-only data,
seed 0. In the ORIGINAL data gender adds nothing to tastes
(dR2 = {dr2_innate:+.4f}). (a) Population dR2 over rounds at beta={FOCUS_BETA:g}.
Gray band: 5-95% of {N_PERM} gender-label permutations per environment at the
final round ([{null_lo:+.4f}, {null_hi:+.4f}]) -- what this estimator returns on
an uninformative feature. Dotted references: the matched no-AI twin (peer
environment, the only one that saves one) and age as a placebo feature; both
stay inside the band throughout. Under full adoption the population never
leaves the band ({v_full:+.4f} at round 29); anchoring lifts it to
{v_anch:+.4f}, and anchoring with peer coupling to {v_peer:+.4f}.
(b) The same quantity for the served model: every environment starts near
{stats[0]['pred'][0]:+.3f} because the beta={FOCUS_BETA:g} round-0 model already
carries gender structure, but full adoption grinds it to {stats[0]['pred'][-1]:+.4f}
as captured opinions collapse onto a few served values, while peer coupling more
than doubles it ({p_peer[0]:+.4f} to {p_peer[-1]:+.4f}). (c) Final-round
population dR2 against beta: nothing clears the null band below
beta={FOCUS_BETA:g}, so endogenization switches on at a HIGHER regularization
than population capture, which turns on between beta=0.2 and 0.5. Every run
here resets its adapter to the pristine snapshot before each round's training,
so the rise cannot be accumulated weight drift -- it is carried by the data the
population generates. Single seed; the effect sizes are of the same order as
the run-to-run spread measured on matched pairs elsewhere, so treat the
ordering, not the magnitudes, as the result.""")


if __name__ == "__main__":
    main()
