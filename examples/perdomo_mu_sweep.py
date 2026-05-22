"""Perdomo Figure 2 reproduction: mu-sweep on the strategic-loan scenario.

Sweeps mus across [0.01, 1, 100, 1000] (Perdomo notebook's canonical
eps_list) for each learner kind (RRM via ERMLearner, RGD via GradientLearner
with one SGD step/round). Plots gap, performative risk, and ||theta|| in a
(n_learners x 3) panel grid.

Saturation handling: when ||theta_t|| > SATURATION_THRESHOLD (1e8) the gap
curve drops out -- at that magnitude LBFGS finds a numerical stationary
point and reports gap=0, which is not meaningful convergence. The
theta-norm panel still shows the divergence.

Defaults to GiveMeSomeCredit via Kaggle; set USE_SYNTHETIC=True in the
config cell to force the synthetic fallback.
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch

    from perfsim.history import History
    from perfsim.scenarios.perdomo_loan.config import PerdomoLoanConfig
    from perfsim.scenarios.perdomo_loan.reproduction import run
    return History, Path, PerdomoLoanConfig, plt, run, torch


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # Perdomo mu sweep

        Sweeps mu over orders of magnitude and tracks: stability gap,
        performative risk, predictor norm. RRM (LBFGS to convergence) and
        RGD (one SGD step/round) compared side by side.

        At large mu RRM saturates numerically (LBFGS settles at a point with
        ||theta|| ~ 1e10+ and reported gap=0). We mask the gap curve past
        that threshold and keep the norm panel as the witness of divergence.
        """
    )
    return (mo,)


@app.cell
def constants():
    DEFAULT_MUS = (0.01, 1.0, 100.0, 1000.0)
    DEFAULT_LEARNERS = ("erm", "gradient")
    LEARNER_TITLES = {
        "erm": "RRM (ERMLearner; LBFGS to convergence)",
        "gradient": "RGD (GradientLearner; one SGD step / round)",
    }
    GAP_FLOOR = 1e-12
    SATURATION_THRESHOLD = 1e8
    return (
        DEFAULT_LEARNERS,
        DEFAULT_MUS,
        GAP_FLOOR,
        LEARNER_TITLES,
        SATURATION_THRESHOLD,
    )


@app.cell
def config():
    MUS = (0.01, 1.0, 100.0, 1000.0)
    LEARNERS = ("erm", "gradient")
    N_ROUNDS = 25
    WEIGHT_DECAY = 5e-5
    RGD_LR = 0.1
    RGD_STEPS = 1
    SEED = 0
    USE_SYNTHETIC = False
    return (
        LEARNERS,
        MUS,
        N_ROUNDS,
        RGD_LR,
        RGD_STEPS,
        SEED,
        USE_SYNTHETIC,
        WEIGHT_DECAY,
    )


@app.cell
def run_sweep_fn(History, PerdomoLoanConfig, run):
    def run_sweep(
        mus, learners,
        *, n_rounds, weight_decay, rgd_lr, rgd_steps, seed, use_synthetic_fallback,
    ):
        out = {ln: {} for ln in learners}
        for ln in learners:
            for mu in mus:
                cfg = PerdomoLoanConfig(
                    mu=mu,
                    n_rounds=n_rounds,
                    learner=ln,
                    learner_lr=rgd_lr,
                    learner_steps=rgd_steps,
                    weight_decay=weight_decay,
                    seed=seed,
                    use_synthetic_fallback=use_synthetic_fallback,
                )
                print(f"  learner={ln:>8s}  mu={mu:>8g}  hash={cfg.content_hash()}  running...")
                out[ln][mu] = run(cfg)
        return out
    return (run_sweep,)


@app.cell
def extract_fn(torch):
    def extract_series_inner(history):
        rounds, prs, gaps, theta_norms = [], [], [], []
        for r in history.records:
            rounds.append(int(r["round"]))
            pr = r.get("PR")
            prs.append(float(pr.item()) if isinstance(pr, torch.Tensor) else float("nan"))
            gap = r.get("stability_gap")
            gaps.append(float(gap.item()) if isinstance(gap, torch.Tensor) else None)
            theta = r.get("theta")
            theta_norms.append(
                float(theta.norm().item()) if isinstance(theta, torch.Tensor) else float("nan")
            )
        return rounds, prs, gaps, theta_norms
    extract_series = extract_series_inner
    return (extract_series,)


@app.cell
def execute(
    LEARNERS,
    MUS,
    N_ROUNDS,
    RGD_LR,
    RGD_STEPS,
    SEED,
    USE_SYNTHETIC,
    WEIGHT_DECAY,
    run_sweep,
):
    results = run_sweep(
        MUS, LEARNERS,
        n_rounds=N_ROUNDS,
        weight_decay=WEIGHT_DECAY,
        rgd_lr=RGD_LR,
        rgd_steps=RGD_STEPS,
        seed=SEED,
        use_synthetic_fallback=USE_SYNTHETIC,
    )
    return (results,)


@app.cell
def summary(SATURATION_THRESHOLD, extract_series, results):
    for _ln, _per_mu in results.items():
        print(f"# learner={_ln}")
        for _mu, _hist in sorted(_per_mu.items()):
            _, _prs, _gaps, _tnorms = extract_series(_hist)
            _final_pr = _prs[-1] if _prs else float("nan")
            _final_gap = next((g for g in reversed(_gaps) if g is not None), None)
            _gap_s = f"{_final_gap:.3e}" if _final_gap is not None else "n/a"
            _final_norm = _tnorms[-1] if _tnorms else float("nan")
            _sat = " [SATURATED]" if _final_norm > SATURATION_THRESHOLD else ""
            print(f"  mu={_mu:>8g}  PR={_final_pr:.6f}  gap={_gap_s}  ||theta||={_final_norm:.3e}{_sat}")
    return


@app.cell
def plot_row_fn(GAP_FLOOR, LEARNER_TITLES, SATURATION_THRESHOLD, extract_series):
    def plot_row_inner(axes, results_for_learner, learner_name, *, is_top_row):
        ax_gap, ax_pr, ax_norm = axes
        for mu, history in sorted(results_for_learner.items()):
            rounds, prs, gaps, theta_norms = extract_series(history)
            label = f"mu={mu:g}"

            gap_x, gap_y = [], []
            for t, g, tn in zip(rounds, gaps, theta_norms):
                if g is None:
                    continue
                if tn > SATURATION_THRESHOLD:
                    break
                gap_x.append(t)
                gap_y.append(max(g, GAP_FLOOR))
            if gap_y:
                ax_gap.plot(gap_x, gap_y, marker="o", markersize=3, label=label)
            else:
                ax_gap.plot([], [], marker="o", markersize=3, label=f"{label} (saturated)")
            ax_pr.plot(rounds, prs, marker="o", markersize=3, label=label)
            ax_norm.plot(rounds, theta_norms, marker="o", markersize=3, label=label)

        ax_gap.set_yscale("log")
        ax_gap.axhline(GAP_FLOOR, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_gap.set_ylabel(f"{LEARNER_TITLES[learner_name]}\n||theta_t - theta_{{t-1}}||")
        if is_top_row:
            ax_gap.set_title(f"Stability gap (mask at ||theta||>{SATURATION_THRESHOLD:g})")
        ax_gap.legend(fontsize=8)
        ax_gap.grid(True, which="both", alpha=0.3)

        ax_pr.set_yscale("symlog", linthresh=1.0)
        ax_pr.set_ylabel("PR (symlog)")
        if is_top_row:
            ax_pr.set_title("PR per round")
        ax_pr.legend(fontsize=8)
        ax_pr.grid(True, which="both", alpha=0.3)

        ax_norm.set_yscale("log")
        ax_norm.axhline(SATURATION_THRESHOLD, color="red", linewidth=0.7, linestyle="--",
                       alpha=0.6, label="saturation")
        ax_norm.set_ylabel("||theta_t|| (log y)")
        if is_top_row:
            ax_norm.set_title("Predictor norm")
        ax_norm.legend(fontsize=8)
        ax_norm.grid(True, which="both", alpha=0.3)
        for ax in (ax_gap, ax_pr, ax_norm):
            ax.set_xlabel("round")
    plot_row = plot_row_inner
    return (plot_row,)


@app.cell
def plot(DEFAULT_LEARNERS, plot_row, plt, results):
    learner_order = [ln for ln in DEFAULT_LEARNERS if ln in results]
    n_rows = len(learner_order)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows), squeeze=False)
    for i, ln in enumerate(learner_order):
        plot_row(axes[i], results[ln], ln, is_top_row=(i == 0))
    fig.suptitle("Perdomo strategic-loan sweep (perfsim)")
    fig.tight_layout()
    fig
    return (axes, fig, learner_order)


if __name__ == "__main__":
    app.run()
