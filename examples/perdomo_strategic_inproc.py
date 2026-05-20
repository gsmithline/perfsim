"""End-to-end demo: Perdomo strategic-loan scenario, single μ, in-process.

Drives the full Perdomo loan scenario through `Simulator.run` without any
agent-shell / Executor indirection. Prints per-round PR and stability gap.

Defaults to GiveMeSomeCredit via Kaggle (cached locally). Pass `--synthetic`
to use the in-process synthetic fallback (no Kaggle creds required, not the
replication claim).

Run:
    python examples/perdomo_strategic_inproc.py
    python examples/perdomo_strategic_inproc.py --mu 100 --n-rounds 25
    python examples/perdomo_strategic_inproc.py --synthetic
"""

from __future__ import annotations

import argparse

import torch

from perfsim.scenarios.perdomo_loan.config import PerdomoLoanConfig
from perfsim.scenarios.perdomo_loan.reproduction import run


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--n-rounds", type=int, default=15)
    p.add_argument("--learner", choices=("erm", "gradient"), default="erm")
    p.add_argument("--learner-lr", type=float, default=0.01)
    p.add_argument("--learner-steps", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    config = PerdomoLoanConfig(
        mu=args.mu,
        n_rounds=args.n_rounds,
        learner=args.learner,
        learner_lr=args.learner_lr,
        learner_steps=args.learner_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_synthetic_fallback=args.synthetic,
    )

    print(
        f"# Perdomo strategic-loan (in-proc): mu={config.mu} "
        f"learner={config.learner} rounds={config.n_rounds} "
        f"weight_decay={config.weight_decay} synthetic={config.use_synthetic_fallback}"
    )
    print(f"# config hash: {config.content_hash()}")
    print(f"# strat_features: {list(config.strat_features)}")

    history = run(config)

    print(f"# {'round':>5}  {'PR':>12}  {'||Δθ||':>12}")
    for r in history.records:
        pr = r.get("PR")
        gap = r.get("stability_gap")
        pr_s = f"{pr.item():12.6f}" if isinstance(pr, torch.Tensor) else f"{'n/a':>12}"
        gap_s = (
            f"{gap.item():12.3e}"
            if isinstance(gap, torch.Tensor)
            else f"{'(init)':>12}"
        )
        print(f"  {r['round']:>5d}  {pr_s}  {gap_s}")

    final_theta = history.records[-1]["theta"]
    print(f"# final ||θ||_2 = {final_theta.norm().item():.6f}")
    print(f"# final theta first 4 entries: {final_theta.flatten()[:4].tolist()}")


if __name__ == "__main__":
    main()
