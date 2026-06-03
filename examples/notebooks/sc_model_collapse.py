import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.feature_selection import mutual_info_classif

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "serif"
    return (
        LogisticRegression,
        Ridge,
        mutual_info_classif,
        np,
        os,
        pd,
        plt,
        preprocessing,
        roc_auc_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Does strategic classification cause model collapse?  (Tier 1)

    ## The idea, plainly

    "Model collapse" is usually told as: a model trains on its own *synthetic* output, and
    quality/diversity decay over generations. The deeper point isn't *who wrote the data*
    (human vs synthetic), it's *what process produced it*: **was the data generated
    independently of the model, or did the model shape it?**

    **Strategic classification is the clean human-data version.** A platform deploys a
    classifier (say, credit scoring). People see it and change their features to get a better
    score. The platform later retrains on those changed, still human-authored, features. So the
    training data is **model-mediated**, not exogenous. The loop:

    $$\theta_t \;\to\; \text{people respond: } x' = R(x,\theta_t) \;\to\; D_t \;\to\; \theta_{t+1}.$$

    **The question this notebook tests:** does recursively retraining on this model-mediated
    human data *collapse* anything — even with no synthetic data added?

    ## What "collapse" could mean (two different things)

    1. **Feature homogenization** — everyone's profile becomes alike (feature variance shrinks).
    2. **Information collapse** — the observed features stop carrying information about the
       *true* label (a credit profile stops predicting actual default). This is the one with teeth.

    ## How people respond (two *derived* best-responses, nothing hand-tuned to collapse)

    - **`shift`** (Perdomo's): everyone nudges their gameable features the same way, toward a
      better score. A uniform translation.
    - **`jump`**: people near the decision cutoff who can afford it hop just over it (move onto
      the boundary, bounded by a cost budget). The other rational response.

    Neither is built to collapse anything — both come from "maximize the chance of a good
    score minus a cost."

    ## Two ways the platform's data evolves

    - **standard PP**: each round people respond *from their true features* to the current
      classifier (the classic repeated-retraining loop).
    - **compounding**: last round's gamed features become this round's starting point (the
      self-consuming version, where model-mediated data piles up).

    And **`replace`** (retrain on gamed data only) vs **`accumulate`** (keep the original real
    data in the pool, an exogenous anchor).

    ## What we'll find (spoiler, and it's honest)

    Under these *derived* responses, **nothing collapses**: information stays flat, features
    stay spread out, the loop just converges to a mildly gaming-degraded stable point — exactly
    what performative-prediction theory says. **Collapse is not implicit in strategic
    classification.** That's the baseline. It tells us the collapse story needs an *explicit*
    extra ingredient — a *shared* assistant (LLM) that makes many people respond the *same*
    way (convergence to a common template) — which is the Tier-2 experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paper code (Perdomo), data + the two derived best-responses
    """)
    return


@app.cell
def _(np, pd, preprocessing):
    def load_data(file_loc):  # from the credit repo
        """Load GiveMeSomeCredit, scale, add a bias column, balance classes."""
        data = pd.read_csv(file_loc, index_col=0)
        data.dropna(inplace=True)
        X_all = preprocessing.scale(data.drop("SeriousDlqin2yrs", axis=1))
        X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)
        Y_all = np.array(data["SeriousDlqin2yrs"])  # 1 = serious delinquency (default)
        default_idx = np.where(Y_all == 1)[0]
        other_idx = np.where(Y_all == 0)[0][:10000]
        idx = np.concatenate((default_idx, other_idx))
        p = np.random.permutation(len(idx))
        return X_all[idx][p], Y_all[idx][p]

    return (load_data,)


@app.cell
def _(np):
    def best_response_shift(X, theta, epsilon, sf):
        """Perdomo: everyone moves gameable features by -epsilon*theta (toward a lower
        predicted-default score). A uniform translation — preserves variance AND separability."""
        Z = np.copy(X)
        Z[:, sf] += -epsilon * theta[sf]
        return Z

    def best_response_jump(X, theta, delta, sf):
        """Cost-capped jump-to-the-boundary. Score s = X @ theta predicts default; favorable = s<0.
        An unfavorable agent (s>0) who can reach the boundary within a feature budget `delta`
        moves along theta exactly onto s=0; others give up. Bounded (feature move = s/||theta_sf||
        <= delta), so it cannot blow up. Derived from linear utility + quadratic cost."""
        s = X @ theta
        Z = np.copy(X)
        tn = float(np.sqrt(np.sum(theta[sf] ** 2))) + 1e-12
        move = (s > 0) & (s <= tn * delta)
        alpha = np.where(move, -s / (tn * tn), 0.0)  # move down to s = 0
        Z[:, sf] += alpha[:, None] * theta[sf][None, :]
        return Z

    return best_response_jump, best_response_shift


@app.cell
def _(np):
    def logistic_regression(X, Y, lam, tol=1e-6, theta_init=None):
        """L2-regularized logistic regression (Perdomo, warm-startable; exp clipped for stability).
        This is the *platform's* classifier."""
        X = np.copy(X)
        Y = np.copy(Y)
        n, d = X.shape
        smooth = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n)
        eta0 = 1 / (smooth + lam)

        def loss(t):
            z = np.clip(X @ t, -30, 30)
            return 1.0 / n * np.sum(-Y * (X @ t) + np.log(1 + np.exp(z))) + lam / 2 * np.linalg.norm(t[:-1]) ** 2

        theta = np.zeros(d) if theta_init is None else np.copy(theta_init)
        prev, gap, eta = loss(theta), 1e30, eta0
        while gap > tol:
            z = np.clip(X @ theta, -30, 30)
            c = np.exp(z) / (1 + np.exp(z)) - Y
            grad = 1.0 / n * np.sum(X * c[:, None], axis=0) + lam * np.append(theta[:-1], 0)
            nt = theta - eta * grad
            ln = loss(nt)
            if ln > prev:
                eta *= 0.1
                gap = 1e30
                continue
            eta = eta0
            theta, gap, prev = nt, prev - ln, ln
        return theta

    return (logistic_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data + the two collapse measures
    """)
    return


@app.cell
def _(load_data, np, os):
    np.random.seed(0)
    _path = os.path.expanduser("~/.cache/perfsim/datasets/GiveMeSomeCredit/cs-training.csv")
    X0, Y = load_data(_path)
    n, d = X0.shape
    GAMEABLE = list(range(10))  # baseline: all 10 features gameable (cleanest null)
    # advisor sections need a non-gameable split (delinquency history + age are not gameable):
    GAME = [0, 3, 4, 5, 7, 9]   # surface features the advisor can recommend
    NONG = [1, 2, 6, 8]         # age + delinquency history (fixed) -> advisor INPUTS
    mi_sub = np.random.choice(n, 3000, replace=False)  # subsample for the (slow) MI estimate
    print(f"n={n}  d={d}  base default rate={Y.mean():.3f}")
    return GAME, GAMEABLE, NONG, X0, Y, mi_sub


@app.cell
def _(GAMEABLE, LogisticRegression, mi_sub, mutual_info_classif, roc_auc_score):
    def info_observed(Xg, Y):
        """INFORMATION COLLAPSE measure #1: the best achievable AUC on the *observed* (gamed) data.
        Refit a fresh classifier on (Xg, Y) -- if gaming destroyed the signal, even the best model
        can't separate and this falls to 0.5. It's a property of the data, not of theta_t."""
        m = LogisticRegression(max_iter=200, C=100).fit(Xg[:, :-1], Y)
        return float(roc_auc_score(Y, m.decision_function(Xg[:, :-1])))

    def info_mi(Xg, Y):
        """INFORMATION COLLAPSE measure #2: total mutual information between the observed gameable
        features and the true label (subsampled). Falls if observed features stop carrying signal."""
        return float(mutual_info_classif(Xg[mi_sub][:, GAMEABLE], Y[mi_sub], random_state=0).sum())

    return info_mi, info_observed


@app.cell
def _():
    LAM = 1.0 / 100   # platform classifier regularization
    N_ROUNDS = 12
    EPS_SHIFT = 5.0   # shift best-response strength
    DELTA = 1.0       # jump best-response feature budget
    return DELTA, EPS_SHIFT, LAM, N_ROUNDS


@app.cell
def _(
    DELTA,
    EPS_SHIFT,
    GAMEABLE,
    LAM,
    N_ROUNDS,
    X0,
    Y,
    best_response_jump,
    best_response_shift,
    info_mi,
    info_observed,
    logistic_regression,
    np,
    roc_auc_score,
):
    def run(br_kind, regime, mode):
        """One full SC retraining loop. y is fixed (gaming, not improvement)."""
        theta = logistic_regression(X0, Y, LAM)
        X = X0.copy()  # carried-forward population (used only in 'compound' mode)
        rows = []
        for t in range(N_ROUNDS):
            base = X if mode == "compound" else X0  # respond from gamed (compound) or true (PP)
            if br_kind == "shift":
                Xg = best_response_shift(base, theta, EPS_SHIFT, GAMEABLE)
            else:
                Xg = best_response_jump(base, theta, DELTA, GAMEABLE)
            accepted = (Xg @ theta) < 0  # platform decides on the OBSERVED (gamed) features
            rows.append(
                {
                    "round": t,
                    "info_observed": info_observed(Xg, Y),       # collapse measure: signal left in data
                    "info_mi": info_mi(Xg, Y),                    # collapse measure: MI(obs feats; y)
                    "strat_std": float(np.mean(np.std(Xg[:, GAMEABLE], axis=0))),  # homogenization
                    "auc_true": float(roc_auc_score(Y, X0 @ theta)),  # theta on TRUE feats (param drift)
                    "def_among_accepted": float(Y[accepted].mean()) if accepted.any() else np.nan,
                    "accept_rate": float(accepted.mean()),
                }
            )
            if regime == "replace":
                trainX, trainY = Xg, Y
            else:  # accumulate: keep the original real data as an exogenous anchor
                trainX, trainY = np.vstack([X0, Xg]), np.concatenate([Y, Y])
            theta = logistic_regression(trainX, trainY, LAM, theta_init=theta)
            if mode == "compound":
                X = Xg

        return rows

    runs = {
        "shift / replace / standard-PP": run("shift", "replace", "standard"),
        "shift / replace / compounding": run("shift", "replace", "compound"),
        "shift / accumulate / compounding": run("shift", "accumulate", "compound"),
        "jump / replace / compounding": run("jump", "replace", "compound"),
    }
    print("done:", list(runs))
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    Watch the two **information-collapse** panels (top row): if collapse were happening they'd
    fall. The homogenization panel (`strat_std`) likewise. `auc_true` is the deployed classifier
    scored on *true* features — it can drift, but that's a parameter artifact, not loss of
    information (compare to `info_observed`, which is the best signal *recoverable* from the data).
    """)
    return


@app.cell
def _(plt, runs):
    def _plot():
        panels = [
            ("info_observed", "INFO COLLAPSE? best AUC on observed data\n(flat = signal preserved)"),
            ("info_mi", "INFO COLLAPSE? MI(observed feats; true y)\n(flat = signal preserved)"),
            ("strat_std", "HOMOGENIZATION? strategic-feature std\n(flat = no homogenization)"),
            ("auc_true", "deployed $\\theta$ on TRUE features\n(parameter drift, not info loss)"),
            ("def_among_accepted", "true default rate among accepted\n(decision harm)"),
            ("accept_rate", "accept rate"),
        ]
        colors = {
            "shift / replace / standard-PP": "C0",
            "shift / replace / compounding": "C3",
            "shift / accumulate / compounding": "C2",
            "jump / replace / compounding": "C1",
        }
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, (key, title) in zip(axes.flat, panels):
            for name, rows in runs.items():
                ax.plot([r["round"] for r in rows], [r[key] for r in rows], "o-", ms=3,
                        color=colors[name], label=name)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("round")
            if key in ("info_observed", "auc_true"):
                ax.axhline(0.5, color="k", ls=":", lw=1)
        axes.flat[0].legend(fontsize=7, loc="lower left")
        fig.suptitle("Does strategic classification collapse model-mediated human data? (GiveMeSomeCredit)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 2 — add a shared advisor (the missing ingredient)

        SC alone didn't collapse. Now introduce a *shared* advisor: everyone is pulled toward a
        common recommended profile ("look like an approved applicant"). The response blends each
        agent's own best-response with the shared template, weighted by **λ = how generic the
        advice is** (input-insensitivity). λ=0 = do-it-yourself (no contraction); λ→1 = everyone
        follows the same template (contraction toward a point). The population contracts at rate
        (1−λ). We sweep λ and watch the gameable-feature spread.
        """
    )
    return


@app.cell
def _(GAME, X0, Y, info_observed, logistic_regression, np):
    EPS_ADV = 5.0
    LAM_REG = 1.0 / 100
    NR_ADV = 10

    def br_blend(X, theta, lam_adv):
        Z = np.copy(X)
        indiv = X[:, GAME] - EPS_ADV * theta[GAME]              # own best-response (shift)
        s = X @ theta
        fav = s < 0
        c = X[fav][:, GAME].mean(axis=0) if fav.any() else X[:, GAME].mean(axis=0)
        Z[:, GAME] = (1.0 - lam_adv) * indiv + lam_adv * c       # blend toward shared template
        return Z

    def run_blend(lam_adv):
        theta = logistic_regression(X0, Y, LAM_REG)
        X = X0.copy()
        std = []
        for _t in range(NR_ADV):
            Xg = br_blend(X, theta, lam_adv)
            std.append(float(np.mean(np.std(Xg[:, GAME], axis=0))))
            theta = logistic_regression(Xg, Y, LAM_REG, theta_init=theta)
            X = Xg
        return {"std": std, "info_final": float(info_observed(Xg, Y))}

    LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    blend = {la: run_blend(la) for la in LAMBDAS}
    print("lambda-blend done:", LAMBDAS)
    return LAMBDAS, blend


@app.cell
def _(LAMBDAS, blend, plt):
    def _p():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for la in LAMBDAS:
            ax[0].plot(blend[la]["std"], "o-", ms=3, label=f"$\\lambda$={la}")
        ax[0].set_xlabel("round")
        ax[0].set_ylabel("gameable-feature std")
        ax[0].set_title("homogenization over rounds, by advice genericness $\\lambda$")
        ax[0].legend(fontsize=8)
        ax[1].plot(LAMBDAS, [blend[la]["std"][-1] for la in LAMBDAS], "o-", color="C3")
        ax[1].set_xlabel("$\\lambda$  (genericness of shared advice)")
        ax[1].set_ylabel("final gameable-feature std")
        ax[1].set_title("COLLAPSE vs $\\lambda$\n(contraction turns on as advice gets generic)")
        fig.tight_layout()
        return fig

    _p()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 3 — make the advisor an actual fitted model (ridge)

        Instead of setting λ by hand, *fit* the advisor: a ridge regressor mapping each agent's
        **fixed** traits (age, delinquency history) to a recommended gameable profile, trained on
        approved applicants. **The ridge penalty is the genericness knob, emergently:** a large
        penalty makes it ignore inputs and predict the mean profile (generic, contractive); a small
        penalty makes it personalized (preserves spread). We *measure* the advisor's
        input-insensitivity as the **dispersion** of its recommendations across agents (low
        dispersion = generic), and check that collapse tracks it. So genericness isn't assumed — it
        falls out of the model's regularization.
        """
    )
    return


@app.cell
def _(GAME, NONG, Ridge, X0, Y, info_observed, logistic_regression, np):
    LAM_REG2 = 1.0 / 100
    NR_R = 10

    def br_ridge(X, theta, alpha):
        s = X @ theta
        fav = s < 0
        Z = np.copy(X)
        if fav.sum() < 10:
            return Z, 0.0
        adv = Ridge(alpha=alpha).fit(X[fav][:, NONG], X[fav][:, GAME])  # learn approved profile
        rec = adv.predict(X[:, NONG])
        unf = np.where(~fav)[0]
        Z[np.ix_(unf, GAME)] = rec[unf]                                  # unfavorable adopt advice
        dispersion = float(np.mean(np.std(rec, axis=0)))                 # low = generic advice
        return Z, dispersion

    def run_ridge(alpha):
        theta = logistic_regression(X0, Y, LAM_REG2)
        X = X0.copy()
        std = []
        disp = float("nan")
        for _t in range(NR_R):
            Xg, disp = br_ridge(X, theta, alpha)
            std.append(float(np.mean(np.std(Xg[:, GAME], axis=0))))
            theta = logistic_regression(Xg, Y, LAM_REG2, theta_init=theta)
            X = Xg
        return {"std": std, "disp": disp, "info_final": float(info_observed(Xg, Y))}

    ALPHAS = [0.01, 1.0, 100.0, 1e4, 1e6]
    ridge = {a: run_ridge(a) for a in ALPHAS}
    print("ridge advisor done:", ALPHAS)
    return ALPHAS, ridge


@app.cell
def _(ALPHAS, plt, ridge):
    def _p():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        disps = [ridge[a]["disp"] for a in ALPHAS]
        finals = [ridge[a]["std"][-1] for a in ALPHAS]
        for a in ALPHAS:
            ax[0].plot(ridge[a]["std"], "o-", ms=3, label=f"$\\alpha$={a:g}")
        ax[0].set_xlabel("round")
        ax[0].set_ylabel("gameable-feature std")
        ax[0].set_title("homogenization over rounds, by ridge penalty")
        ax[0].legend(fontsize=8)
        ax[1].plot(disps, finals, "o-", color="C3")
        for a, xx, yy in zip(ALPHAS, disps, finals):
            ax[1].annotate(f"$\\alpha$={a:g}", (xx, yy), fontsize=7)
        ax[1].set_xlabel("advice dispersion across agents  (low = generic, EMERGENT)")
        ax[1].set_ylabel("final gameable-feature std")
        ax[1].set_title("COLLAPSE vs EMERGENT genericness\n(a generic fitted advisor contracts the population)")
        fig.tight_layout()
        return fig

    _p()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this shows (and why it matters)

    **No collapse, under any derived response.**
    - `info_observed` and `info_mi` stay flat: the observed data keeps all its signal about true
      default. A fresh classifier refit on the gamed data is just as good. **No information collapse.**
    - `strat_std` stays flat: **no feature homogenization.** (A uniform `shift` is a translation;
      `jump` only nudges a thin band onto the boundary.)
    - The loop converges to a mildly gaming-degraded **stable point**, exactly as
      performative-prediction theory predicts. `def_among_accepted` takes a one-shot hit from
      gaming and then sits flat, it does not run away.
    - `auc_true` (the deployed $\theta$ scored on *true* features) does drift, but that's the
      classifier's parameters moving, not the data losing information — `info_observed` shows the
      signal is fully recoverable. So it is not a collapse.

    **The takeaway:** *strategic classification, on its own, does not cause model collapse.*
    Rational individual gaming + retraining just reaches a stable point with the data still
    informative. This is the honest baseline, and it matches the literature (PP frames the fixed
    point as convergence, not collapse).

    **So what would cause collapse?** An *explicit* contraction — many people responding the
    **same** way. The natural source is a **shared assistant (LLM)**: if everyone asks the same
    model how to adapt, responses converge to a common template, which *is* a contraction toward
    a shared point. That is the missing ingredient, and the **Tier-2** experiment: an
    LLM-graded / LLM-assisted text setting where you *measure* whether shared LLM advice makes
    the response contractive (template convergence) and whether *that* finally collapses the
    distribution.

    **Caveats.** Single dataset/seed; `y` fixed (gaming, not improvement); the platform retrains
    on gamed features with true labels; vary EPS_SHIFT / DELTA / LAM / GAMEABLE to probe robustness.
    """)
    return


if __name__ == "__main__":
    app.run()
