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

    from sklearn.mixture import GaussianMixture

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "serif"
    return GaussianMixture, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-consuming performative loops: stability vs collapse

    **Claim under test.** A retraining loop is a distribution map
    $D(\theta)$ with a performatively stable fixed point. When the loop is
    *anchored* by a fraction $\alpha$ of exogenous (real) data it converges to
    a healthy fixed point with a positive variance floor. When it is *closed*
    ($\alpha \to 0$, the model retrains purely on its own finite samples) the
    same map's stable point is a **degenerate point mass** — i.e. model
    collapse. Model collapse is what performative stability looks like in the
    closed-loop regime, and $\alpha$ is the knob between the two.

    **The honest mechanism.** Non-expansiveness of $D$ (PP's $\varepsilon$-sensitivity)
    gives convergence in $\theta$-space — it does *not* by itself shrink
    variance. Collapse is a **finite-sample** effect: each generation refits on
    $N$ draws, and the variance estimate is biased low / the tails thin out;
    with no exogenous anchor there is no restoring force. So the precise
    statement is **closed loop + finite-sample refitting ⇒ strict contraction
    to a point mass**, and the experiment must vary exactly two things:
    the exogenous fraction $\alpha$ and the per-generation sample size $N$.

    **Gaussian recursion (provable core).** Draw $N$ samples each generation
    from $$D = \alpha\,\mathcal N(\mu_{\text{real}}, \sigma_{\text{real}}^2)
    + (1-\alpha)\,\mathcal N(\hat\mu, \hat\sigma^2)$$ and refit by MLE. With
    means aligned, the fitted variance obeys

    $$\mathbb{E}[\hat\sigma^2_{t+1}] = \Big(1-\tfrac1N\Big)\Big[\alpha\,\sigma_{\text{real}}^2 + (1-\alpha)\,\hat\sigma^2_t\Big],
    \qquad
    \sigma^2_\infty = \frac{(1-\tfrac1N)\,\alpha\,\sigma_{\text{real}}^2}{1-(1-\tfrac1N)(1-\alpha)}.$$

    $\alpha\to0 \Rightarrow \sigma^2_\infty\to0$ (collapse, rate $1-\tfrac1N$);
    $\alpha>0 \Rightarrow$ positive floor; $N\to\infty \Rightarrow \sigma_{\text{real}}^2$
    (the population map does **not** collapse).
    """)
    return


@app.cell
def _(GaussianMixture, np):
    def run_gaussian_loop(
        alpha,
        N,
        n_gen,
        sigma_real=1.0,
        mu_real=0.0,
        sigma0=1.0,
        mu0=0.0,
        ddof=0,           # 0 = MLE (biased low), 1 = unbiased (Bessel)
        tail_c=2.0,       # tail mass threshold, in units of sigma_real
        seed=0,
    ):
        """One trajectory of the anchored self-consuming 1-D Gaussian loop.

        Each generation draws N points from the mixture
            alpha * N(mu_real, sigma_real^2) + (1-alpha) * N(mu_hat, sigma_hat^2)
        and refits (mu_hat, sigma_hat^2) on the drawn sample. alpha=0 is the pure
        closed (self-consuming) loop. Returns per-generation variance, mean, and
        empirical tail mass P(|x - mu_real| > tail_c * sigma_real).
        """
        rng = np.random.default_rng(seed)
        mu_hat = mu0
        var_hat = sigma0 ** 2

        var_h = [var_hat]
        mu_h = [mu_hat]
        tail_h = [np.nan]

        for _ in range(n_gen):
            from_real = rng.random(N) < alpha
            n_real = int(from_real.sum())

            x = np.empty(N)
            x[from_real] = rng.normal(mu_real, sigma_real, n_real)
            sd = np.sqrt(var_hat) if var_hat > 0 else 0.0
            x[~from_real] = rng.normal(mu_hat, sd, N - n_real)

            mu_hat = x.mean()
            var_hat = x.var(ddof=ddof)

            var_h.append(var_hat)
            mu_h.append(mu_hat)
            tail_h.append(np.mean(np.abs(x - mu_real) > tail_c * sigma_real))

        return np.array(var_h), np.array(mu_h), np.array(tail_h)


    def run_many(alpha, N, n_gen, n_seeds, which=0, **kw):
        """Stack `which`-th output (0=var, 1=mu, 2=tail) over n_seeds trajectories."""
        rows = [
            run_gaussian_loop(alpha, N, n_gen, seed=s, **kw)[which]
            for s in range(n_seeds)
        ]
        return np.vstack(rows)


    def variance_recursion(alpha, N, n_gen, v0, sigma_real=1.0):
        """Deterministic expected-variance recursion (the analytical prediction)."""
        v = v0
        hist = [v]
        for _ in range(n_gen):
            v = (1.0 - 1.0 / N) * (alpha * sigma_real ** 2 + (1.0 - alpha) * v)
            hist.append(v)
        return np.array(hist)


    def variance_floor(alpha, N, sigma_real=1.0):
        """Closed-form fixed point of the expected-variance recursion."""
        num = (1.0 - 1.0 / N) * alpha * sigma_real ** 2
        den = 1.0 - (1.0 - 1.0 / N) * (1.0 - alpha)
        return num / den


    def run_gmm_loop(alpha, N, n_gen, K, means_real, sigma_real=0.5, seed=0):
        """Self-consuming K-component GMM loop. Tracks sorted component weights and
        the number of 'alive' modes (weight > 1/(2K)) per generation. alpha mixes
        in fresh real draws each generation."""
        rng = np.random.default_rng(seed)
        means_real = np.asarray(means_real, dtype=float)

        def sample_real(m):
            comp = rng.integers(0, K, m)
            return rng.normal(means_real[comp], sigma_real)

        x = sample_real(N)
        weights_h = []
        n_modes_h = []

        for g in range(n_gen):
            gm = GaussianMixture(
                K, covariance_type="spherical", n_init=2, random_state=int(seed * 1000 + g)
            )
            gm.fit(x.reshape(-1, 1))

            weights_h.append(np.sort(gm.weights_)[::-1].copy())
            n_modes_h.append(int(np.sum(gm.weights_ > 1.0 / (2 * K))))

            n_real = int((rng.random(N) < alpha).sum())
            xs_model = gm.sample(max(N - n_real, 1))[0].ravel()[: N - n_real]
            xs_real = sample_real(n_real)
            x = np.concatenate([xs_real, xs_model])
            rng.shuffle(x)

        return np.array(weights_h), np.array(n_modes_h)

    return (
        run_gaussian_loop,
        run_gmm_loop,
        run_many,
        variance_floor,
        variance_recursion,
    )


@app.cell
def _():
    # knobs
    SIGMA_REAL = 1.0
    N_DEFAULT = 30      # samples per generation for the alpha sweep
    N_GEN = 120         # generations
    N_SEEDS = 60        # trajectories to average

    ALPHAS = [0.0, 0.02, 0.1, 0.3]          # exogenous fractions (0 = closed loop)
    N_LIST = [10, 30, 100, 300]             # sample sizes for the rate sweep

    K = 4                                   # GMM components
    MEANS_REAL = [-6.0, -2.0, 2.0, 6.0]     # well-separated modes
    GMM_N = 50                              # samples/generation for the GMM loop
    GMM_NGEN = 60
    GMM_SEEDS = 15
    return (
        ALPHAS,
        GMM_N,
        GMM_NGEN,
        GMM_SEEDS,
        K,
        MEANS_REAL,
        N_DEFAULT,
        N_GEN,
        N_LIST,
        N_SEEDS,
        SIGMA_REAL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1 — the exogenous fraction $\alpha$ sets the variance floor

    Fixed $N$, sweep $\alpha$. Solid = mean over seeds; dashed = analytical
    recursion; dotted horizontal = closed-form floor $\sigma^2_\infty$.
    The $\alpha=0$ curve drives to zero; any $\alpha>0$ parks at a floor.
    """)
    return


@app.cell
def _(
    ALPHAS,
    N_DEFAULT,
    N_GEN,
    N_SEEDS,
    SIGMA_REAL,
    run_many,
    variance_floor,
    variance_recursion,
):
    exp1 = {"alphas": ALPHAS, "mean": [], "pred": [], "floor": []}
    for _a in ALPHAS:
        _V = run_many(_a, N_DEFAULT, N_GEN, N_SEEDS, which=0, sigma_real=SIGMA_REAL, sigma0=SIGMA_REAL)
        exp1["mean"].append(_V.mean(axis=0))
        exp1["pred"].append(variance_recursion(_a, N_DEFAULT, N_GEN, SIGMA_REAL ** 2, SIGMA_REAL))
        exp1["floor"].append(variance_floor(_a, N_DEFAULT, SIGMA_REAL))
    return (exp1,)


@app.cell
def _(N_DEFAULT, exp1, plt):
    def _plot():
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis([0.1, 0.4, 0.65, 0.9])
        for i, a in enumerate(exp1["alphas"]):
            ax.plot(exp1["mean"][i], color=colors[i], lw=2.5,
                    label=r"$\alpha$ = {} (sim)".format(a))
            ax.plot(exp1["pred"][i], color=colors[i], lw=1.2, ls="--")
            if exp1["floor"][i] > 0:
                ax.axhline(exp1["floor"][i], color=colors[i], lw=0.8, ls=":")
        ax.set_title(r"Variance vs generation ($N={}$)".format(N_DEFAULT), fontsize=15)
        ax.set_xlabel("Generation $t$", fontsize=13)
        ax.set_ylabel(r"fitted variance $\hat\sigma^2_t$", fontsize=13)
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.text(0.98, 0.04, "dashed = analytical recursion · dotted = floor",
                transform=ax.transAxes, ha="right", fontsize=9, color="0.4")
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2 — sample size $N$ sets the collapse rate (closed loop, $\alpha=0$)

    Larger $N$ slows collapse (rate $1-\tfrac1N$); the population limit
    $N\to\infty$ would not collapse at all.
    """)
    return


@app.cell
def _(N_GEN, N_LIST, N_SEEDS, SIGMA_REAL, run_many, variance_recursion):
    exp2 = {"Ns": N_LIST, "mean": [], "pred": []}
    for _N in N_LIST:
        _V = run_many(0.0, _N, N_GEN, N_SEEDS, which=0, sigma_real=SIGMA_REAL, sigma0=SIGMA_REAL)
        exp2["mean"].append(_V.mean(axis=0))
        exp2["pred"].append(variance_recursion(0.0, _N, N_GEN, SIGMA_REAL ** 2, SIGMA_REAL))
    return (exp2,)


@app.cell
def _(exp2, plt):
    def _plot():
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.plasma([0.1, 0.4, 0.65, 0.9])
        for i, N in enumerate(exp2["Ns"]):
            ax.plot(exp2["mean"][i], color=colors[i], lw=2.5, label=r"$N$ = {} (sim)".format(N))
            ax.plot(exp2["pred"][i], color=colors[i], lw=1.2, ls="--")
        ax.set_title(r"Closed-loop collapse, $\alpha=0$", fontsize=15)
        ax.set_xlabel("Generation $t$", fontsize=13)
        ax.set_ylabel(r"fitted variance $\hat\sigma^2_t$", fontsize=13)
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.text(0.98, 0.04, "dashed = analytical $(1-1/N)^t$ decay",
                transform=ax.transAxes, ha="right", fontsize=9, color="0.4")
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 3 — collapse happens even with the *unbiased* estimator

    With Bessel's correction the variance is a martingale ($\mathbb{E}[\hat\sigma^2_{t+1}]=\hat\sigma^2_t$,
    flat black line), yet individual closed-loop trajectories still collapse
    to zero almost surely — because variance evolves *multiplicatively* and
    $\mathbb{E}[\log(\chi^2_{N-1}/(N-1))]<0$ (Jensen). So collapse is not just
    MLE bias.
    """)
    return


@app.cell
def _(N_GEN, SIGMA_REAL, np, run_gaussian_loop):
    _N3 = 20
    exp3 = {"N": _N3, "traj": []}
    for _s in range(25):
        _v, _, _ = run_gaussian_loop(0.0, _N3, N_GEN, sigma_real=SIGMA_REAL,
                                     sigma0=SIGMA_REAL, ddof=1, seed=100 + _s)
        exp3["traj"].append(_v)
    exp3["traj"] = np.vstack(exp3["traj"])
    return (exp3,)


@app.cell
def _(SIGMA_REAL, exp3, plt):
    def _plot():
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in exp3["traj"]:
            ax.plot(row, color="C0", lw=0.8, alpha=0.4)
        ax.plot(exp3["traj"].mean(axis=0), color="C3", lw=2.5, label="mean over seeds")
        ax.axhline(SIGMA_REAL ** 2, color="k", lw=1.2, ls="--",
                   label=r"$\mathbb{E}[\hat\sigma^2]$ (martingale)")
        ax.set_title(r"Unbiased estimator, closed loop ($N={}$)".format(exp3["N"]), fontsize=15)
        ax.set_xlabel("Generation $t$", fontsize=13)
        ax.set_ylabel(r"fitted variance $\hat\sigma^2_t$", fontsize=13)
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 4 — GMM: closed-loop self-consumption drops modes

    Where "mode collapse" is literal. A $K$-component GMM refit on its own
    samples loses components (a mode underrepresented in one finite sample is
    underrepresented in the next, and dies). An exogenous fraction $\alpha>0$
    keeps the modes alive.
    """)
    return


@app.cell
def _(GMM_N, GMM_NGEN, GMM_SEEDS, K, MEANS_REAL, np, run_gmm_loop):
    exp4 = {"alphas": [0.0, 0.1], "modes": {}, "weights": {}}
    for _a in exp4["alphas"]:
        _modes = []
        _w_one = None
        for _s in range(GMM_SEEDS):
            _w, _m = run_gmm_loop(_a, GMM_N, GMM_NGEN, K, MEANS_REAL, sigma_real=0.5, seed=_s)
            _modes.append(_m)
            if _s == 0:
                _w_one = _w
        exp4["modes"][_a] = np.vstack(_modes)
        exp4["weights"][_a] = _w_one
    return (exp4,)


@app.cell
def _(K, exp4, plt):
    def _plot():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # (left) mean number of alive modes vs generation
        ax = axes[0]
        for a in exp4["alphas"]:
            m = exp4["modes"][a]
            ax.plot(m.mean(axis=0), lw=2.5, label=r"$\alpha$ = {}".format(a))
        ax.axhline(K, color="0.5", lw=0.8, ls=":")
        ax.set_title("Surviving modes vs generation", fontsize=14)
        ax.set_xlabel("Generation $t$", fontsize=12)
        ax.set_ylabel(r"# components with weight $> 1/2K$", fontsize=12)
        ax.set_ylim(0, K + 0.4)
        ax.legend(fontsize=11)

        ax = axes[1]
        w = exp4["weights"][0.0]
        for k in range(w.shape[1]):
            ax.plot(w[:, k], lw=2, label=r"comp {}".format(k + 1))
        ax.set_title(r"Component weights, one run ($\alpha=0$)", fontsize=14)
        ax.set_xlabel("Generation $t$", fontsize=12)
        ax.set_ylabel("sorted weight", fontsize=12)
        ax.legend(fontsize=10)

        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Exp 1–2:** the closed-form recursion matches the simulation. $\alpha$
      sets the variance floor ($\to 0$ as $\alpha\to 0$); $N$ sets the rate
      ($1-\tfrac1N$). The population map ($N\to\infty$) does not collapse —
      collapse is finite-sample.
    - **Exp 3:** collapse survives bias correction (multiplicative martingale,
      negative log-drift), so it is structural, not an artifact of MLE bias.
    - **Exp 4:** in a mixture, "collapse" shows up as **mode death**; $\alpha$
      anchoring preserves modes.

    This is the bare contraction/collapse picture with no $\beta\!\cdot\!\mathrm{KL}$ —
    the anchor lives entirely in $\alpha$ (the $\alpha\!\cdot\! D_{\text{real}}$
    mixture is the data-mixture form of the KL-to-real restoring force).
    Next: swap the Gaussian/GMM refit for an LM and re-measure variance/tail/
    perplexity against these $\alpha$- and $N$-predictions.
    """)
    return


if __name__ == "__main__":
    app.run()
