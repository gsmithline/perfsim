"""Block 1: static benchmark. Fixed uniform population, no movement.

Establishes the competitive equilibrium structure before any feedback:
  - landscape: best response to two fixed rivals. Does a platform want to
    spread (sit in the empty arc) or clump (sit next to a rival)?
  - basins: from many random starts, how often K platforms segment vs clump.

Findings: under softmax choice the best response is to spread (peak at the
equidistant spot); under hard nearest choice the payoff is flat. The circle has
multiple equilibria (segmented and paired). Basins are parity-dependent: K=3
always segments, K=4 mostly pairs, K=5 mostly segments.
"""

import torch

from experiments.competition._common import (
    classify,
    hard_share_landscape,
    opt_run,
    softmax_share_landscape,
)


def landscape():
    rivals = [0.0, 0.33]
    print("LANDSCAPE (circle): rivals at 0.0 and 0.33, sweep the third platform.")
    print("equidistant third = 0.665. does captured share peak there?\n")
    cand, sh = hard_share_landscape(rivals)
    print(
        f"  hard nearest      argmax c={float(cand[sh.argmax()]):.3f}"
        f"  flat?(max-median)={float(sh.max() - sh.median()):.4f}"
    )
    for tau in [0.2, 0.1, 0.05, 0.02]:
        cand, sh = softmax_share_landscape(rivals, tau)
        print(
            f"  softmax tau={tau:<5} argmax c={float(cand[sh.argmax()]):.3f}"
            f"  flat?(max-median)={float(sh.max() - sh.median()):.4f}"
        )


def basins(n_starts=24, tau=0.2):
    print(f"\nBASINS (circle, tau={tau}): random starts -> segment or clump?\n")
    for k in [3, 4, 5]:
        counts = {"equidistant": 0, "clumped": 0, "other": 0}
        for seed in range(n_starts):
            g = torch.Generator().manual_seed(1000 + seed)
            inits = torch.rand(k, generator=g).tolist()
            _, gaps = opt_run(k, inits, tau=tau, seed=seed)
            counts[classify(k, gaps)] += 1
        print(
            f"  K={k}: equidistant {counts['equidistant']:2d} | "
            f"clumped {counts['clumped']:2d} | other {counts['other']:2d}  (of {n_starts})"
        )


if __name__ == "__main__":
    landscape()
    basins()
