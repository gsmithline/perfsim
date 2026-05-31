import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "serif"
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Lock-in Hypothesis as a performative loop

    Reproduction of the formal Bayesian model from **Qiu, He, Chugh &
    Kleiman-Weiner, *The Lock-in Hypothesis: Stagnation by Algorithm* (ICML
    2025)** — [arXiv:2506.06166](https://arxiv.org/abs/2506.06166) — set in
    perfsim's performative-prediction frame.

    **Model (their §3).** $N$ agents (agent 0 = the AI/LLM, agents $1..N{-}1$ =
    humans) estimate an unknown truth $\mu$. Each step every agent takes a
    fresh private noisy measurement $o_{i,t}\sim\mathcal N(\mu,\sigma^2)$ and
    maintains a private posterior $(\hat\mu_i, p_i)$ and an *aggregate* belief
    $(\hat\nu_i, q_i)$. A trust matrix $W$ says how much each agent defers to
    others' aggregate beliefs; the AI aggregates everyone and broadcasts back.
    Transition dynamics (their eqs 3–6), with $\odot$ = elementwise product:

    $$p_{t+1}=p_t+\sigma^{-2}\mathbf 1,\quad
    \hat\mu_{t+1}\odot p_{t+1}=\hat\mu_t\odot p_t+\sigma^{-2}o_t,$$
    $$q_{t+1}=p_{t+1}+W q_t,\quad
    \hat\nu_{t+1}\odot q_{t+1}=\hat\mu_{t+1}\odot p_{t+1}+W(\hat\nu_t\odot q_t).$$

    **Lock-in (their Thm 3.2 / Cor 3.3).** With the human–LLM trust matrix
    (AI trusts each human by $\lambda_1$, each human trusts the AI by
    $\lambda_2$, no human–human edges), $\rho(W)=\sqrt{(N{-}1)\lambda_1\lambda_2}$.
    If $\rho(W)>1$ the aggregate beliefs **lock in to a false, over-confident
    consensus** ($\Pr[\lim\hat\nu=\mu]=0$); if $\rho(W)<1$ they converge to the
    truth. The phase transition is at $(N{-}1)\lambda_1\lambda_2=1$.

    **Performative reading.** The broadcast AI aggregate is the *deployed model*
    $\theta_t$; it shapes the population's next-round beliefs (the *data*
    $D_t$); the AI re-aggregates (*retrains*). Lock-in is a **performatively
    stable point that is degenerate** — the population stops moving under
    retraining but has collapsed onto an over-confident, possibly false
    consensus. The fresh private measurements $o_t$ are an *exogenous anchor*
    (the analogue of an $\alpha\!\cdot\!D_{\text{real}}$ mixture or KL-to-real
    force). Critically: **that anchor never stops — full, unlimited fresh
    evidence arrives every round — yet lock-in still happens once $\rho(W)>1$,**
    because the trust/mediation feedback amplifies beliefs geometrically while
    evidence accumulates only linearly. That is a concrete instance of the
    thesis that *infinite real data fails to anchor a sufficiently strong
    mode-seeking / high-trust mediation channel*.
    """)
    return


@app.cell
def _(np):
    def build_W(N, lam1, lam2):
        """Human-LLM trust matrix (their Example 3.1 / Cor 3.3). Agent 0 = AI:
        trusts each human by lam1. Agents 1..N-1 = humans: each trusts only the AI
        by lam2. No human-human edges. rho(W) = sqrt((N-1) lam1 lam2)."""
        W = np.zeros((N, N))
        W[0, 1:] = lam1
        W[1:, 0] = lam2
        return W


    def simulate(N, lam1, lam2, mu=1.0, sigma=1.0, T=120, seed=0):
        """Run the Bayesian lock-in dynamics (their eqs 3-6) for T steps.

        Returns per-step (T, N) arrays of aggregate beliefs nu and aggregate
        precisions q. Carries the natural parameters mp = muhat*p and nq = nuhat*q.
        """
        rng = np.random.default_rng(seed)
        W = build_W(N, lam1, lam2)
        inv_var = sigma ** -2

        p = np.zeros(N)
        mp = np.zeros(N)   # private:   mp = muhat * p
        q = np.zeros(N)
        nq = np.zeros(N)   # aggregate: nq = nuhat * q

        nu_hist = np.empty((T, N))
        q_hist = np.empty((T, N))
        for t in range(T):
            o = rng.normal(mu, sigma, N)
            p = p + inv_var                 # (3)
            mp = mp + inv_var * o           # (4)
            q = p + W @ q                   # (5)
            nq = mp + W @ nq                # (6)
            nu_hist[t] = nq / q
            q_hist[t] = q
        return nu_hist, q_hist


    def lam_for_target(target, N):
        """Symmetric lam1=lam2 giving (N-1) lam^2 = target (so rho = sqrt(target))."""
        return np.sqrt(target / (N - 1))

    return build_W, lam_for_target, simulate


@app.cell
def _():
    # knobs
    MU = 1.0            # ground truth
    SIGMA = 1.0         # measurement noise
    N = 11              # 1 AI + 10 humans  -> (N-1) = 10
    T = 120             # time steps
    SEEDS = 15          # independent runs (matches the paper's figure)

    TARGETS = [0.9, 1.0, 1.1]                  # (N-1)l1l2 for the phase-change figure
    SWEEP = [0.4, 0.6, 0.8, 0.9, 0.95, 1.0,
             1.05, 1.1, 1.25, 1.5, 2.0, 3.0]   # for the transition curve
    SEEDS_SWEEP = 30
    return MU, N, SEEDS, SEEDS_SWEEP, SIGMA, SWEEP, T, TARGETS


@app.cell
def _(N, build_W, lam_for_target, np):
    # sanity check: rho(W) matches sqrt((N-1) lam1 lam2)
    for _tg in [0.5, 1.0, 1.5]:
        _lam = lam_for_target(_tg, N)
        _rho = max(abs(np.linalg.eigvals(build_W(N, _lam, _lam))))
        print(f"(N-1)l1l2={_tg}:  rho(W)={_rho:.4f}   sqrt(target)={np.sqrt(_tg):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — the phase change (reproduction)

    Top: collective belief $\overline{\hat\nu}_t$ across seeds (black = truth
    $\mu$). Bottom: collective posterior entropy. Below threshold beliefs
    converge to $\mu$ with calibrated uncertainty; above it they lock onto
    different false values while becoming over-confident (entropy plummeting).
    """)
    return


@app.cell
def _(MU, N, SEEDS, SIGMA, T, TARGETS, lam_for_target, np, simulate):
    phase = []
    for _tg in TARGETS:
        _lam = lam_for_target(_tg, N)
        _belief = np.empty((SEEDS, T))
        _entropy = np.empty((SEEDS, T))
        for _s in range(SEEDS):
            _nu, _q = simulate(N, _lam, _lam, mu=MU, sigma=SIGMA, T=T, seed=_s)
            _belief[_s] = _nu.mean(axis=1)                       # collective belief
            _qbar = _q.mean(axis=1)
            _entropy[_s] = 0.5 * np.log(2 * np.pi * np.e / _qbar)  # collective entropy
        phase.append({"target": _tg, "belief": _belief, "entropy": _entropy})
    return (phase,)


@app.cell
def _(MU, phase, plt):
    def _plot():
        ncol = len(phase)
        fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 7), squeeze=False)
        for c, res in enumerate(phase):
            ax = axes[0][c]
            for row in res["belief"]:
                ax.plot(row, color="C0", lw=1.0, alpha=0.5)
            ax.axhline(MU, color="k", lw=1.2, ls="--", label="truth $\\mu$")
            ax.set_title(r"$(N{-}1)\lambda_1\lambda_2 = $" + str(res["target"]), fontsize=14)
            ax.set_ylabel("collective belief $\\overline{\\hat\\nu}_t$", fontsize=12)
            if c == 0:
                ax.legend(fontsize=10)

            ax = axes[1][c]
            ax.plot(res["entropy"].mean(axis=0), color="C3", lw=2.5)
            ax.set_xlabel("step $t$", fontsize=12)
            ax.set_ylabel("collective posterior entropy", fontsize=12)
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — the transition: error and over-confidence vs trust

    Sweeping $(N{-}1)\lambda_1\lambda_2$ through the critical value 1. Left:
    error of the final consensus from the truth. Right: final posterior std
    (over-confidence) on a log scale. Both turn sharply at the dashed line
    $\rho(W)=1$ — the performative-stability/lock-in boundary.
    """)
    return


@app.cell
def _(MU, N, SEEDS_SWEEP, SIGMA, SWEEP, T, lam_for_target, np, simulate):
    sweep = {"target": list(SWEEP), "rmse": [], "post_std": []}
    for _tg in SWEEP:
        _lam = lam_for_target(_tg, N)
        _finals = np.empty(SEEDS_SWEEP)
        _stds = np.empty(SEEDS_SWEEP)
        for _s in range(SEEDS_SWEEP):
            _nu, _q = simulate(N, _lam, _lam, mu=MU, sigma=SIGMA, T=T, seed=_s)
            _finals[_s] = _nu[-1].mean()
            _stds[_s] = (1.0 / np.sqrt(_q[-1])).mean()
        sweep["rmse"].append(np.sqrt(np.mean((_finals - MU) ** 2)))
        sweep["post_std"].append(_stds.mean())
    return (sweep,)


@app.cell
def _(plt, sweep):
    def _plot():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = sweep["target"]

        ax = axes[0]
        ax.plot(x, sweep["rmse"], "o-", color="C0", lw=2.5)
        ax.axvline(1.0, color="k", ls="--", lw=1.2, label=r"$\rho(W)=1$")
        ax.set_xlabel(r"$(N{-}1)\lambda_1\lambda_2$", fontsize=13)
        ax.set_ylabel(r"error of consensus from truth", fontsize=13)
        ax.set_title("Lock-in to a false value", fontsize=14)
        ax.legend(fontsize=11)

        ax = axes[1]
        ax.plot(x, sweep["post_std"], "o-", color="C3", lw=2.5)
        ax.axvline(1.0, color="k", ls="--", lw=1.2, label=r"$\rho(W)=1$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$(N{-}1)\lambda_1\lambda_2$", fontsize=13)
        ax.set_ylabel(r"final posterior std (over-confidence)", fontsize=13)
        ax.set_title("Diversity / uncertainty collapse", fontsize=14)
        ax.legend(fontsize=11)

        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation & next step

    - **Lock-in = degenerate performative stability.** Above $\rho(W)=1$ the
      retraining loop converges (it's stable) but onto an over-confident, false
      consensus — collapse, not health. Below it, the same loop tracks the truth.
    - **The anchor here is unlimited and still fails.** Fresh, truth-bearing
      private measurements arrive *every* round (an infinite-data anchor), yet
      $\rho(W)>1$ locks the population to falsehood — because mediated belief
      compounds geometrically ($\rho^t$) while evidence accumulates linearly
      ($p_t = t\sigma^{-2}$). This is the concrete realization of regime (ii):
      *quantity of real signal cannot rescue a strong mediation channel.*
    - **Maps to our knobs.** $\rho(W)=\sqrt{(N{-}1)\lambda_1\lambda_2}$ is the
      mediation/trust gain — the role $\lambda$ (override strength) plays in the
      Gaussian toy. Raising it past 1 is the analogue of the mode-seeking
      channel defeating the anchor.

    **Next:** add an explicit exogenous-anchor / discounting knob (e.g. let
    humans down-weight the AI by a factor, or inject a fraction of agents who
    ignore the broadcast) to trace the diversity floor as a function of trust —
    the same $(\alpha, \lambda)$ sweep as `gaussian_self_consuming.py`, now on
    the Lock-in dynamics. Then port to their LLM-agent / WildChat testbed.
    """)
    return


if __name__ == "__main__":
    app.run()
