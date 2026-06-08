"""Matrix-level analysis of a competition/hunt run: treat each platform as a
point in R^N (its per-agent prediction vector) instead of a scalar mean.
Reads a saved trajectory.pt (keys: preds_raw [T,P,N], op_raw [T,N], innate [N]).

Run: python experiments/analyze_matrix.py <run_dir_or_trajectory.pt> [more ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def load(path: Path) -> dict:
    p = path / "trajectory.pt" if path.is_dir() else path
    d = torch.load(p, map_location="cpu", weights_only=False)
    return {"name": p.parent.name, "preds": d["preds_raw"].float(),
            "op": d["op_raw"].float(), "innate": d["innate"].float()}


def series(run: dict) -> dict:
    preds, op, innate = run["preds"], run["op"], run["innate"]   # (T,P,N),(T,N),(N,)
    T, P, N = preds.shape
    inter, to_truth, to_op = [], [], []
    for t in range(T):
        F = preds[t]                                             # (P, N)
        pairs = [(F[a] - F[b]).abs().mean().item()
                 for a in range(P) for b in range(a + 1, P)]
        inter.append(sum(pairs) / len(pairs))
        to_truth.append([(F[p] - innate).abs().mean().item() for p in range(P)])
        to_op.append([(F[p] - op[t]).abs().mean().item() for p in range(P)])
    return {"inter": torch.tensor(inter),
            "to_truth": torch.tensor(to_truth),                  # (T, P)
            "to_op": torch.tensor(to_op), "P": P, "N": N}


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or [Path("runs/pokec_fj_hunt/hunt3_t05")]
    runs = [load(p) for p in paths]
    out = Path("runs/analysis")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cmap = plt.cm.tab10.colors

    for r, run in enumerate(runs):
        s = series(run)
        c = cmap[r % 10]
        axes[0].plot(s["inter"], color=c, label=run["name"])
        # mean over platforms of distance-to-truth and distance-to-current-opinion
        axes[1].plot(s["to_truth"].mean(dim=1), color=c, label=run["name"])
        axes[2].plot(s["to_op"].mean(dim=1), color=c, label=run["name"])

    axes[0].set_title("inter-platform distance  mean_pairs ||f_p - f_q||_1 / N\n(the merge, full matrix)")
    axes[1].set_title("distance to INNATE truth  ||f_p - x*||_1 / N\n(near truth vs manufactured attractor)")
    axes[2].set_title("distance to CURRENT opinion  ||f_p - x||_1 / N")
    for ax in axes:
        ax.set_xlabel("round")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7)

    fig.suptitle("Matrix-level competition analysis (platforms as points in R^N)",
                 fontweight="bold")
    fig.savefig(out / "matrix_distances.png", dpi=150)

    print(f"{'run':>24s} | {'inter f->l':>14s} {'to_truth f->l':>16s}")
    for run in runs:
        s = series(run)
        it = f"{s['inter'][0]:.3f}->{s['inter'][-1]:.3f}"
        tt = f"{s['to_truth'].mean(1)[0]:.3f}->{s['to_truth'].mean(1)[-1]:.3f}"
        print(f"{run['name']:>24s} | {it:>14s} {tt:>16s}")
    print(f"[analysis] figure -> {out / 'matrix_distances.png'}")


if __name__ == "__main__":
    main()
