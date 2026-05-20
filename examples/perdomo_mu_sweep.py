"""
Perdomo Figure 2 reproduction: μ-sweep on the strategic-loan scenario.

Runs the Perdomo loan scenario across mus = [0.01, 1, 100, 1000] (Perdomo
notebook's canonical eps_list), collects PR, stability_gap, and ||θ|| per
round, and saves a three-panel plot:

(left)   ||θ_t - θ_{t-1}||  log y    — Perdomo Fig 2(a) analogue
(middle) PR(θ_t)            symlog y — loss trajectory
(right)  ||θ_t||            log y    — exposes numerical overflow

After ||θ_t|| >  default 1e8 the gap curve is
dropped from the stability-gap panel: at that magnitude LBFGS finds a
numerical stationary point and reports gap=0, which is not convergence to
a meaningful fixed point. The θ-norm panel still shows the saturated value
so the divergence is visible.

Defaults to the real GiveMeSomeCredit data via KaggleDataset, passes --synthetic 
to force the synthetic fallback (smoke test only, not the replication claim).

Run:
    python -m examples.perdomo_mu_sweep
    python examples/perdomo_mu_sweep.py --synthetic --n-rounds 15
    python examples/perdomo_mu_sweep.py --mus 0.01 1 100 1000 --out fig.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


import matplotlib.pyplot as plt
import torch

from perfsim.history import History
from perfsim.scenarios.perdomo_loan.config import PerdomoLoanConfig
from perfsim.scenarios.perdomo_loan.reproduction import run

DEFAULT_MUS = (0.01, 1.0, 100.0, 1000.0)


def run_sweep(
    mus: tuple[float, ...],
    *,
    n_rounds: int = 25,
    learner: str = "erm",
    weight_decay: float = 5e-5,
    seed: int = 0,
    use_synthetic_fallback: bool = False,
) -> dict[float, History]:
    """Run the Perdomo scenario for each μ; return {μ: History}."""
    out: dict[float, History] = {}
    for mu in mus:
        config = PerdomoLoanConfig(
            mu=mu,
            n_rounds=n_rounds,
            learner=learner,
            weight_decay=weight_decay,
            seed=seed,
            use_synthetic_fallback=use_synthetic_fallback,
        )
        print(f"  μ={mu:>8g}  config_hash={config.content_hash()}  running...")
        history = run(config)
        out[mu] = history
    return out


def _extract_series(
    history: History,
) -> tuple[list[int], list[float], list[float | None], list[float]]:
    rounds: list[int] = []
    prs: list[float] = []
    gaps: list[float | None] = []
    theta_norms: list[float] = []
    for r in history.records:
        rounds.append(int(r["round"]))
        pr = r.get("PR")
        prs.append(float(pr.item()) if isinstance(pr, torch.Tensor) else float("nan"))
        gap = r.get("stability_gap")
        if isinstance(gap, torch.Tensor):
            gaps.append(float(gap.item()))
        else:
            gaps.append(None)
        theta = r.get("theta")
        if isinstance(theta, torch.Tensor):
            theta_norms.append(float(theta.norm().item()))
        else:
            theta_norms.append(float("nan"))
    return rounds, prs, gaps, theta_norms


GAP_FLOOR = 1e-12 
SATURATION_THRESHOLD = 1e8

def plot_sweep(results: dict[float, History], out_path: Path) -> None:
    fig, (ax_gap, ax_pr, ax_norm) = plt.subplots(1, 3, figsize=(15, 4.5))
    for mu, history in sorted(results.items()):
        rounds, prs, gaps, theta_norms = _extract_series(history)
        label = f"μ={mu:g}"

        gap_x: list[int] = []
        gap_y: list[float] = []
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
            ax_gap.plot([], [], marker="o", markersize=3,
                        label=f"{label} (saturated)")

        ax_pr.plot(rounds, prs, marker="o", markersize=3, label=label)
        ax_norm.plot(rounds, theta_norms, marker="o", markersize=3, label=label)

    ax_gap.set_yscale("log")
    ax_gap.axhline(GAP_FLOOR, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_gap.set_xlabel("round t")
    ax_gap.set_ylabel(r"$\|\theta_t - \theta_{t-1}\|_2$ (clamped at $10^{-12}$)")
    ax_gap.set_title(rf"Stability gap (masked at $\|\theta\|>{SATURATION_THRESHOLD:g}$)")
    ax_gap.legend()
    ax_gap.grid(True, which="both", alpha=0.3)

    ax_pr.set_yscale("symlog", linthresh=1.0)
    ax_pr.set_xlabel("round t")
    ax_pr.set_ylabel("performative risk (symlog)")
    ax_pr.set_title("PR per round")
    ax_pr.legend()
    ax_pr.grid(True, which="both", alpha=0.3)

    ax_norm.set_yscale("log")
    ax_norm.axhline(
        SATURATION_THRESHOLD, color="red", linewidth=0.7, linestyle="--",
        alpha=0.6, label="saturation threshold",
    )
    ax_norm.set_xlabel("round t")
    ax_norm.set_ylabel(r"$\|\theta_t\|_2$ (log y)")
    ax_norm.set_title("Predictor norm")
    ax_norm.legend()
    ax_norm.grid(True, which="both", alpha=0.3)

    fig.suptitle("Perdomo strategic-loan sweep (perfsim)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  saved figure -> {out_path}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mus", type=float, nargs="+", default=list(DEFAULT_MUS))
    p.add_argument("--n-rounds", type=int, default=25)
    p.add_argument("--learner", choices=("erm", "gradient"), default="erm")
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "perdomo_mu_sweep.png",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    print(f"# Perdomo μ-sweep: mus={args.mus} learner={args.learner} "
          f"n_rounds={args.n_rounds} synthetic={args.synthetic}")
    results = run_sweep(
        tuple(args.mus),
        n_rounds=args.n_rounds,
        learner=args.learner,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_synthetic_fallback=args.synthetic,
    )
    for mu, history in sorted(results.items()):
        _, prs, gaps, theta_norms = _extract_series(history)
        final_pr = prs[-1] if prs else float("nan")
        final_gap = next((g for g in reversed(gaps) if g is not None), None)
        gap_str = f"{final_gap:.3e}" if final_gap is not None else "n/a"
        final_norm = theta_norms[-1] if theta_norms else float("nan")
        sat_flag = " [SATURATED]" if final_norm > SATURATION_THRESHOLD else ""
        print(f"  μ={mu:>8g}  final PR={final_pr:.6f}  final gap={gap_str}  "
              f"|| theta ||={final_norm:.3e}{sat_flag}")
    plot_sweep(results, args.out)


if __name__ == "__main__":
    main()
