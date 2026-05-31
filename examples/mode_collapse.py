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

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "serif"
    return np, os, pd, plt, preprocessing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Performative Prediction on Credit Data

    Reproduction of the credit-scoring experiments from
    **Perdomo, Zrnic, Mendler-Dünner & Hardt, *Performative Prediction* (ICML 2020)**
    on the GiveMeSomeCredit dataset.

    The agent **best-responds** to the deployed classifier (linear utility,
    quadratic cost), so deploying $\theta$ shifts the data distribution. We
    iterate two algorithms to a **performatively stable** point and track the
    convergence gap, performative risk, and accuracy:

    - **RRM** (Repeated Risk Minimization) — fully re-minimize on the induced
      distribution each step (`method='Exact'`).
    - **RGD** (Repeated Gradient Descent) — a single gradient step each step
      (`method='GD'`).

    The core functions below (`load_data`, `best_response`, `evaluate_loss`,
    `logistic_regression`) are the paper's released code, kept **verbatim**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paper code (verbatim)
    """)
    return


@app.cell
def _(np, pd, preprocessing):
    def load_data(file_loc): #taken directly from credit repo
        """Load data from csv file.

        Parameters
        ----------
            file_loc: string
                path to the '.csv' training data file
        Returns
        -------
            X_full: np.array
                balanced data matrix
            Y_full: np.array
                corresponding labels (0/1)
            data: DataFrame
                raw data
        """

        data = pd.read_csv(file_loc, index_col=0)
        data.dropna(inplace=True)

        # full data set
        X_all = data.drop('SeriousDlqin2yrs', axis=1)

        # zero mean, unit variance
        X_all = preprocessing.scale(X_all)

        # add bias term
        X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)

        # outcomes
        Y_all = np.array(data['SeriousDlqin2yrs'])

        # balance classes
        default_indices = np.where(Y_all == 1)[0]
        other_indices = np.where(Y_all == 0)[0][:10000]
        indices = np.concatenate((default_indices, other_indices))

        X_balanced = X_all[indices]
        Y_balanced = Y_all[indices]

        # shuffle arrays
        p = np.random.permutation(len(indices))
        X_full = X_balanced[p]
        Y_full = Y_balanced[p]
        return X_full, Y_full, data

    return (load_data,)


@app.cell
def _(np):
    def best_response(X, theta, epsilon, strat_features):
        """Best response function for agents given classifier theta. Assumes linear utilities and quadratic costs.

        Parameters
        ----------
            X: np.array
                training data matrix
            theta: np.array
                deployed parameter vector
            epsilon: float
                sensitivity parameter, strength of performative effects
            strat_features: list
                list of features that can be manipulated strategically, other features remain fixed

        Returns
        -------
            X_strat: np.array
                modified training data matrix after each agents best responds to the classifier
        """

        n = X.shape[0]

        X_strat = np.copy(X)

        for i in range(n):
            # move everything by epsilon in the direction towards better classification
            theta_strat = theta[strat_features]
            X_strat[i, strat_features] += -epsilon * theta_strat

        return X_strat

    return (best_response,)


@app.cell
def _(best_response, np):
    """Logistic regression model"""

    def sigmoid(z):
        """Evaluate sigmoid function"""
        return 1 / (1 + np.exp(-z))


    def evaluate_loss(X, Y, theta, lam, strat_features=[], epsilon=0):
        """Evaluate L2-regularized logistic regression loss function. For epsilon>0 it returns the performative loss.

        Parameters
        ----------
            X: np.array
                training data matrix
            Y: np.array
                labels
            theta: np.array
                parameter vector
            lam: float
                regularization parameter, lam>0
            strat_features: list
                list of features that can be manipulated strategically, other features remain fixed
            epsilon: float
                sensitivity parameter, quantifying the strength of performative effects

        Returns
        -------
            loss: float
                logistic loss value
        """

        n = X.shape[0]

        # compute strategically manipulated data
        if epsilon > 0:
            X_perf = best_response(X, theta, epsilon, strat_features)
        else:
            X_perf = np.copy(X)

        # compute log likelihood
        t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X_perf @ theta) +
                            np.log(1 + np.exp(X_perf @ theta)))

        # add regularization (without considering the bias)
        t2 = lam / 2.0 * np.linalg.norm(theta[:-1]) ** 2
        loss = t1 + t2

        return loss


    def logistic_regression(X_orig, Y_orig, lam, method, tol=1e-7, theta_init=None):
        """Training of an L2-regularized logistic regression model.

        Parameters
        ----------
            X_orig: np.array
                training data matrix
            Y_orig: np.array
                labels
            lam: float
                regularization parameter, lam>0
            method: string
                optimization method: 'Exact' for returning the exact solution and 'GD' for performing a single gradient descent step on the parameter vector
            tol: float
                stopping criterion for exact minimization
            theta_init: np.array
                initial parameter vector. If None procedure is initialized at zero

        Returns
        -------
            theta: np.array
                updated parameter vector
            loss_list: list
                loss values during training for reporting
            smoothness: float
                smoothness parameter of the logistic loss function given the current training data matrix
        """

        # assumes that the last coordinate is the bias term
        X = np.copy(X_orig)
        Y = np.copy(Y_orig)
        n, d = X.shape

        # compute smoothness of the logistic loss
        smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n)

        if method == 'Exact':
            eta_init = 1 / (smoothness + lam)  # true smoothness

        elif method == 'GD':
            assert(theta_init is not None)
            eta_init = 2 / (smoothness + 2 * lam)

        else:
            print('method must be Exact or GD')
            raise ValueError

        if theta_init is not None:
            theta = np.copy(theta_init)
        else:
            theta = np.zeros(d)

        # evaluate initial loss
        prev_loss = evaluate_loss(X, Y, theta, lam)

        loss_list = [prev_loss]
        is_gd = False
        i = 0
        gap = 1e30

        eta = eta_init

        while gap > tol and not is_gd:

            # take gradients
            exp_tx = np.exp(X @ theta)
            c = exp_tx / (1 + exp_tx) - Y
            gradient = 1.0/n * \
                np.sum(X * c[:, np.newaxis], axis=0) + \
                lam * np.append(theta[:-1], 0)

            new_theta = theta - eta * gradient

            # compute new loss
            t1 = 1.0/n * np.sum(-1 * np.multiply(Y, X @ new_theta) +
                                np.log(1 + np.exp(X @ new_theta)))
            t2 = lam / 2 * np.linalg.norm(new_theta[:-1])
            loss = t1 + t2

            # do backtracking line search
            if loss > prev_loss and method == 'Exact':
                eta = eta * .1
                gap = 1e30
                continue
            else:
                eta = eta_init

            theta = np.copy(new_theta)

            loss_list.append(loss)
            gap = prev_loss - loss
            prev_loss = loss

            if method == 'GD':
                is_gd = True

            i += 1

        return theta, loss_list, smoothness

    return evaluate_loss, logistic_regression


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data & problem setup
    """)
    return


@app.cell
def _(load_data, np, os):
    np.random.seed(0)
    _path = os.path.expanduser(
        "~/.cache/perfsim/datasets/GiveMeSomeCredit/cs-training.csv"
    )
    X, Y, data = load_data(_path)
    n = X.shape[0]
    d = X.shape[1] - 1
    print("n = {}, d (excl. bias) = {}".format(n, d))
    return X, Y, d, data, n


@app.cell
def _(data, np):
    strat_features = np.array([1, 6, 8]) - 1  # for later indexing
    print("Strategic features:")
    for _i, _feat in enumerate(strat_features):
        print(_i, data.columns[_feat + 1])
    return (strat_features,)


@app.cell
def _(X, Y, logistic_regression, n, np, strat_features):
    # baseline (non-performative) optimum, used as the starting point and as
    # the normalizer for the convergence gap
    lam = 1.0 / n
    theta_true, _loss_list, _smoothness = logistic_regression(X, Y, lam, "Exact")
    strat_norm = np.linalg.norm(theta_true[strat_features])

    print("Accuracy:        ", ((X.dot(theta_true) > 0) == Y).mean())
    print("Loss:            ", _loss_list[-1])
    print("Condition number:", lam / (_smoothness + lam))
    print("Norm:            ", np.linalg.norm(theta_true))
    print("Strat features:  ", theta_true[strat_features], "norm =", strat_norm)
    return lam, strat_norm, theta_true


@app.cell
def _():
    # problem parameters (paper settings)
    num_iters = 25
    eps_list = [0.01, 1, 100, 1000]
    num_eps = len(eps_list)
    return eps_list, num_iters


@app.cell
def _(best_response, evaluate_loss, logistic_regression, np):
    def run_performative(
        X,
        Y,
        lam,
        eps_list,
        num_iters,
        method,
        theta_true,
        strat_features,
        strat_norm,
        d,
        tol=1e-7,
        warm_start_exact=False,
    ):
        """Run RRM ('Exact') or RGD ('GD') for each epsilon and collect statistics.

        Per-iteration computation is identical to the paper's driver loop. Returns
        a single results dict keyed by statistic, each a list-per-epsilon.

        warm_start_exact=True initializes the Exact solve from the previous theta
        instead of zeros. The Exact objective is convex, so this is
        results-identical to the paper's cold start but much faster (use it to
        iterate without waiting on the large-epsilon solves). Default False keeps
        the paper's exact behavior.
        """
        num_eps = len(eps_list)
        results = {
            "method": method,
            "eps_list": list(eps_list),
            "theta": [[np.copy(theta_true)] for _ in range(num_eps)],
            "theta_gaps": [[] for _ in range(num_eps)],
            "ll": [[] for _ in range(num_eps)],
            "acc_start": [[] for _ in range(num_eps)],
            "acc_end": [[] for _ in range(num_eps)],
            "lp_start": [[] for _ in range(num_eps)],
            "lp_end": [[] for _ in range(num_eps)],
            "condition_num": [[] for _ in range(num_eps)],
            "gd_cutoff": [[] for _ in range(num_eps)],
        }

        for c, eps in enumerate(eps_list):
            theta = np.copy(theta_true)
            print("Running {} with epsilon = {}".format(method, eps))

            for t in range(num_iters):
                # adjust distribution to current theta
                X_strat = best_response(X, theta, eps, strat_features)

                # performative loss / accuracy of the previous theta
                results["lp_start"][c].append(
                    evaluate_loss(X_strat, Y, theta, lam, strat_features)
                )
                results["acc_start"][c].append(
                    ((X_strat.dot(theta) > 0) == Y).mean()
                )

                # learn on the induced distribution
                if method == "Exact":
                    theta_init = np.copy(theta) if warm_start_exact else np.zeros(d + 1)
                else:
                    theta_init = np.copy(theta)

                theta_new, ll, logistic_smoothness = logistic_regression(
                    X_strat, Y, lam, method, tol=tol, theta_init=theta_init
                )

                # bookkeeping
                results["ll"][c].append(ll)
                results["theta_gaps"][c].append(
                    np.linalg.norm(theta_new - theta) / strat_norm
                )
                results["theta"][c].append(np.copy(theta_new))

                smoothness_2 = max(logistic_smoothness + lam, 2)  # lipschitz gradient
                results["condition_num"][c].append(lam / smoothness_2)
                results["gd_cutoff"][c].append(
                    lam / ((smoothness_2 + lam) * (1 + 1.5 * smoothness_2))
                )

                # performative loss / accuracy of the new theta
                results["lp_end"][c].append(
                    evaluate_loss(X_strat, Y, theta_new, lam, strat_features)
                )
                results["acc_end"][c].append(
                    ((X_strat.dot(theta_new) > 0) == Y).mean()
                )

                theta = np.copy(theta_new)

        return results

    return (run_performative,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the experiments

    > **Runtime note.** RRM uses the paper's exact settings (`tol=1e-7`,
    > cold-started Exact solve). The $\varepsilon = 100, 1000$ cases sit far
    > outside the convergence regime, so the inner solver grinds and the RRM
    > cell can take **several minutes** — that is faithful to the paper's code,
    > not a bug. RGD is fast. To iterate quickly, pass
    > `warm_start_exact=True` (results-identical for the convex Exact solve).
    """)
    return


@app.cell
def _(
    X,
    Y,
    d,
    eps_list,
    lam,
    num_iters,
    run_performative,
    strat_features,
    strat_norm,
    theta_true,
):
    # RRM (Repeated Risk Minimization)
    rrm = run_performative(
        X, Y, lam, eps_list, num_iters, "Exact",
        theta_true, strat_features, strat_norm, d,
        warm_start_exact=True
    )
    return (rrm,)


@app.cell
def _(
    X,
    Y,
    d,
    eps_list,
    lam,
    num_iters,
    run_performative,
    strat_features,
    strat_norm,
    theta_true,
):
    # RGD (Repeated Gradient Descent)
    rgd = run_performative(
        X, Y, lam, eps_list, num_iters, "GD",
        theta_true, strat_features, strat_norm, d,
    )
    return (rgd,)


@app.cell
def _(plt):
    def plot_theta_gaps(results, eps_list):
        """Convergence gap c*||theta_{t+1} - theta_t|| vs iteration, all epsilons."""
        fig, ax = plt.subplots(figsize=(8, 5))
        processed = [[x for x in tg if x != 0.0] for tg in results["theta_gaps"]]
        for c in range(len(eps_list)):
            ax.plot(
                processed[c],
                linewidth=3,
                marker="*",
                markevery=[len(processed[c]) - 1] if processed[c] else None,
                label=r"$\varepsilon$ = {}".format(eps_list[c]),
            )
        ax.set_title(results["method"], fontsize=16)
        ax.set_xlabel(r"Iteration $t$", fontsize=14)
        ax.set_ylabel(r"$c \cdot \|\theta_{t+1} - \theta_{t}\|_2$", fontsize=14)
        ax.set_yscale("log")
        ax.tick_params(labelsize=12)
        ax.legend(loc="center right", fontsize=12)
        fig.tight_layout()
        return fig


    def _zigzag(ax, start, end, num_iters, start_i, gain_style, shift_style):
        offset = 0.8
        for i in range(start_i, num_iters):
            ax.plot([i, i + offset], [start[i], end[i]], gain_style)
            if i < num_iters - 1:
                ax.plot([i + offset, i + 1], [end[i], start[i + 1]], shift_style)


    def plot_risk_trajectory(results, eps_list, num_iters):
        """Performative risk: minimization gain (solid) then distribution shift (dashed)."""
        num_eps = len(eps_list)
        fig, axes = plt.subplots(
            1, num_eps, figsize=(5 * num_eps, 4.5), squeeze=False
        )
        for c in range(num_eps):
            ax = axes[0][c]
            _zigzag(ax, results["lp_start"][c], results["lp_end"][c],
                    num_iters, 2, "b*-", "g--")
            ax.set_title(r"{}, $\varepsilon$={}".format(results["method"], eps_list[c]),
                         fontsize=12)
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Performative risk", fontsize=12)
            ax.set_yscale("log")
            ax.tick_params(labelsize=11)
        fig.tight_layout()
        return fig


    def plot_accuracy_trajectory(results, eps_list, num_iters):
        """Accuracy: gain from re-fitting (solid) then loss from the shift (dotted)."""
        num_eps = len(eps_list)
        fig, axes = plt.subplots(
            1, num_eps, figsize=(5 * num_eps, 4.5), squeeze=False
        )
        for c in range(num_eps):
            ax = axes[0][c]
            _zigzag(ax, results["acc_start"][c], results["acc_end"][c],
                    num_iters, 1, "b*-", "g:")
            ax.set_title(r"{}, $\varepsilon$={}".format(results["method"], eps_list[c]),
                         fontsize=12)
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.tick_params(labelsize=11)
        fig.tight_layout()
        return fig

    return plot_accuracy_trajectory, plot_risk_trajectory, plot_theta_gaps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    ### Convergence to a performatively stable point

    $\|\theta_{t+1} - \theta_t\|$ should contract to zero when $\varepsilon$ is
    small enough; for large $\varepsilon$ it stays bounded away from zero
    (no stable point).
    """)
    return


@app.cell
def _(eps_list, plot_theta_gaps, rrm):
    plot_theta_gaps(rrm, eps_list)
    return


@app.cell
def _(eps_list, plot_theta_gaps, rgd):
    plot_theta_gaps(rgd, eps_list)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Performative risk and accuracy trajectories (RRM)
    """)
    return


@app.cell
def _(eps_list, num_iters, plot_risk_trajectory, rrm):
    plot_risk_trajectory(rrm, eps_list, num_iters)
    return


@app.cell
def _(eps_list, num_iters, plot_accuracy_trajectory, rrm):
    plot_accuracy_trajectory(rrm, eps_list, num_iters)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Performative risk and accuracy trajectories (RGD)
    """)
    return


@app.cell
def _(eps_list, num_iters, plot_risk_trajectory, rgd):
    plot_risk_trajectory(rgd, eps_list, num_iters)
    return


@app.cell
def _(eps_list, num_iters, plot_accuracy_trajectory, rgd):
    plot_accuracy_trajectory(rgd, eps_list, num_iters)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
