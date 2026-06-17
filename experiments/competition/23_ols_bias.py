"""Block 23: Gerstgrasser Figure 7, plus a bias term.

Reproduces the linear-OLS recursion from "Is Model Collapse Inevitable?" (Sec 3,
Thm 1: w_n = w* + X^+ sum_i E_i/i) and asks the question from our discussion:
their accumulate result is bounded (sum 1/i^2 = Basel) *because the noise is
mean-zero*. What happens to accumulate when the per-round synthetic step has a
systematic bias (the performative pull our model has)?

Two bias models, because they need not behave alike:
  additive : Y_next = X w_hat + noise + b           (constant directional drift)
  modepull : Y_next = (1-kappa) X w_hat + kappa*m0 + noise   (shrink toward a mode)

Controls: replace vs accumulate, mean-zero (reproduces the paper), and an
accumulate-with-fixed-pristine-fraction arm (always keep rho real data).

Figure follows the paper: top = replace, bottom = accumulate, run to 2000
model-fitting iterations.

Run: MPLCONFIGDIR=/tmp/mpl python experiments/competition/23_ols_bias.py
"""

import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = 10
T = 100               # samples per iteration (T >= d+2 for ridgeless)
SIGMA = 1.0
N_ITER = 2000
SEEDS = 20
B_ADD = 0.30          # additive drift per round
KAPPA = 0.20          # mode-pull strength
M0 = 3.0              # mode the platform pulls predictions toward (true mean ~0)
RHO = 0.5             # fixed pristine fraction


def run(mode, bias, seed, pristine=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, D))
    w_star = rng.standard_normal(D) / np.sqrt(D)
    Xplus = np.linalg.pinv(X)                       # (X^T X)^-1 X^T
    real = X @ w_star + SIGMA * rng.standard_normal(T)   # block 0 = real data
    sum_blocks = real.copy()                        # running sum (O(1)/iter)
    last = real
    n_blocks = 1
    errs = np.empty(N_ITER)
    for t in range(N_ITER):
        if mode == "replace":
            target = last
        elif pristine and n_blocks > 1:
            syn_mean = (sum_blocks - real) / (n_blocks - 1)
            target = RHO * real + (1 - RHO) * syn_mean
        else:                                       # accumulate
            target = sum_blocks / n_blocks
        w_hat = Xplus @ target
        errs[t] = np.sum((w_hat - w_star) ** 2)
        pred = X @ w_hat
        noise = SIGMA * rng.standard_normal(T)
        if bias == "additive":
            nxt = pred + noise + B_ADD
        elif bias == "modepull":
            nxt = (1 - KAPPA) * pred + KAPPA * M0 + noise
        else:
            nxt = pred + noise
        sum_blocks = sum_blocks + nxt
        last = nxt
        n_blocks += 1
    return errs


def avg(mode, bias, pristine=False):
    return np.mean([run(mode, bias, s, pristine) for s in range(SEEDS)], axis=0)


def main():
    rep = avg("replace", None)
    acc = {
        "accumulate (mean-zero)":             ("#2ca02c", avg("accumulate", None)),
        "accumulate + additive bias":         ("#d62728", avg("accumulate", "additive")),
        "accumulate + mode-pull":             ("#ff7f0e", avg("accumulate", "modepull")),
        "accum + mode-pull + pristine 50%":   ("#1f77b4", avg("accumulate", "modepull", True)),
    }
    basel = SIGMA ** 2 * D / (T - D - 1) * (np.pi ** 2 / 6)
    print(f"d={D} T={T} sigma={SIGMA} N={N_ITER} | Basel bound = {basel:.3f}")
    print(f"{'curve':<36}{'@50':>8}{'@500':>8}{'@2000':>8}")
    print(f"{'replace (mean-zero)':<36}{rep[49]:>8.2f}{rep[499]:>8.2f}{rep[-1]:>8.2f}")
    for k, (_, v) in acc.items():
        print(f"{k:<36}{v[49]:>8.3f}{v[499]:>8.3f}{v[-1]:>8.3f}")

    x = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax[0].plot(x, rep, c="#9467bd", lw=1.5, label="replace (mean-zero)")
    ax[0].set(title="Replace: error grows unboundedly (Gerstgrasser Fig 7 top)",
              ylabel="test error  ||w - w*||^2")
    ax[0].legend(frameon=False, fontsize=8)
    for k, (c, v) in acc.items():
        ax[1].plot(x, v, c=c, lw=1.5, label=k)
    ax[1].axhline(basel, ls="--", c="gray", lw=1, label=f"Basel bound ({basel:.2f})")
    ax[1].set(title="Accumulate: bounded for mean-zero, creeps up under bias, "
                    "re-bounded by fixed pristine fraction",
              xlabel="model-fitting iteration", ylabel="test error  ||w - w*||^2")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Linear OLS recursion, {N_ITER} iterations  (d={D}, T={T}, sigma={SIGMA})")
    fig.savefig("experiments/competition/figs/ols_bias.png", dpi=130)
    print("saved experiments/competition/figs/ols_bias.png")


if __name__ == "__main__":
    main()
